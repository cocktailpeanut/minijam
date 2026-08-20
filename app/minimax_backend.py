from __future__ import annotations

import base64
import copy
import io
import json
import os
import random
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
import soundfile as sf


MAX_SEED = 2_147_483_647
DEFAULT_STEPS = 30
LOW_VRAM_STEPS = 20
GUIDANCE = 1.7
TOP_K = 50


@dataclass(frozen=True)
class GenerationResult:
    wav_bytes: bytes
    seed: int
    duration: float
    mode: str
    steps: int
    tiled_decode: bool


def _system_memory_gb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total_physical / (1024 ** 3)
            return None
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return (page_size * page_count) / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        return None


def _resolve_seed(seed: int | float | str | None) -> int:
    try:
        value = int(seed) if seed is not None else -1
    except (TypeError, ValueError):
        value = -1
    if value < 0:
        return random.randint(0, MAX_SEED)
    return min(value, MAX_SEED)


def _wav_duration(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else 0.0
    except (wave.Error, EOFError):
        return 0.0


def _to_wav(audio_bytes: bytes) -> bytes:
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return audio_bytes
    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        out = io.BytesIO()
        sf.write(out, audio, sample_rate, format="WAV", subtype="PCM_16")
        return out.getvalue()
    except Exception as exc:
        raise RuntimeError("MiniMax returned audio that could not be decoded") from exc


class MiniMaxBackend:
    def __init__(self, workflow_path: str | Path):
        self.engine_url = os.environ.get("MINIMAX_ENGINE_URL", "").strip().rstrip("/")
        self.engine_type = os.environ.get("MINIMAX_ENGINE", "comfyui").strip().lower()
        if self.engine_type not in {"comfyui", "audiocpp"}:
            raise RuntimeError(f"Unsupported MiniMax engine: {self.engine_type}")
        if not self.engine_url:
            raise RuntimeError("MINIMAX_ENGINE_URL is missing; start the app through Pinokio")
        self.workflow_path = Path(workflow_path)
        self.session = requests.Session()
        self.lock = threading.Lock()
        self._engine_info: dict[str, Any] = {}
        self._engine_info_at = 0.0

    def _json(self, method: str, path: str, *, timeout: float = 30, **kwargs: Any) -> Any:
        try:
            response = self.session.request(method, f"{self.engine_url}{path}", timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(f"The MiniMax engine is unavailable at {self.engine_url}") from exc
        if response.status_code >= 400:
            detail = response.text.strip()
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    detail = str(parsed.get("message") or parsed.get("error") or detail)
            except ValueError:
                pass
            raise RuntimeError(detail or f"MiniMax engine request failed ({response.status_code})")
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("The MiniMax engine returned an unreadable response") from exc

    def engine_info(self, *, refresh: bool = False) -> dict[str, Any]:
        if not refresh and self._engine_info and time.monotonic() - self._engine_info_at < 30:
            return dict(self._engine_info)
        if self.engine_type == "audiocpp":
            try:
                response = self.session.get(f"{self.engine_url}/health", timeout=8)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"The MiniMax engine is unavailable at {self.engine_url}") from exc
            info = {
                "engine": "audiocpp",
                "label": "MiniMax Music 3 · Q4 · audio.cpp",
                "total_vram": None,
            }
        else:
            stats = self._json("GET", "/system_stats", timeout=8)
            devices = stats.get("devices", []) if isinstance(stats, dict) else []
            device = devices[0] if devices and isinstance(devices[0], dict) else {}
            name = str(device.get("name") or "GPU").replace("cuda:0 ", "")
            info = {
                "engine": "comfyui",
                "label": f"MiniMax Music 3 · INT8 · {name}",
                "total_vram": device.get("vram_total"),
            }
        self._engine_info = info
        self._engine_info_at = time.monotonic()
        return dict(info)

    def unload_for_composer(self, requested_mode: str) -> None:
        ram_gb = _system_memory_gb()
        should_unload = requested_mode == "low-vram" or ram_gb is None or ram_gb < 40
        if not should_unload:
            return
        try:
            if self.engine_type == "audiocpp":
                response = self.session.post(
                    f"{self.engine_url}/v1/tasks/unload_models",
                    timeout=60,
                    json={"model_ids": ["minimax-music3"]},
                )
            else:
                response = self.session.post(
                    f"{self.engine_url}/free",
                    timeout=60,
                    json={"unload_models": True, "free_memory": True},
                )
            response.raise_for_status()
            print(f"[minimax] unloaded engine before local composer (system_ram={ram_gb})")
        except Exception as exc:
            print(f"[minimax] engine unload skipped: {exc}")

    def _resolve_mode(self, requested: str, duration: float) -> tuple[str, int, bool]:
        value = (requested or "auto").strip().lower()
        if value not in {"auto", "low-vram", "quality"}:
            value = "auto"

        if self.engine_type == "audiocpp":
            if value == "quality":
                return "quality", 20, False
            return ("low-vram" if value == "low-vram" else "auto"), 15, False

        info = self.engine_info()
        raw_vram = info.get("total_vram")
        try:
            vram_gb = float(raw_vram) / (1024 ** 3)
        except (TypeError, ValueError):
            vram_gb = None

        low_hardware = vram_gb is not None and vram_gb < 12
        tiled = value == "low-vram" or duration >= 120 or low_hardware
        steps = LOW_VRAM_STEPS if value == "low-vram" or low_hardware else DEFAULT_STEPS
        if value == "quality":
            steps = DEFAULT_STEPS
            tiled = duration >= 120 or (vram_gb is not None and vram_gb < 10)
        resolved = "low-vram" if value == "auto" and low_hardware else value
        return resolved, steps, tiled

    def _workflow(self, caption: str, lyrics: str, duration: float, seed: int, steps: int, tiled: bool) -> dict[str, Any]:
        try:
            template = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("The bundled MiniMax workflow is unavailable") from exc
        workflow = copy.deepcopy(template)
        workflow["4"]["inputs"].update(
            caption=caption,
            lyrics=lyrics,
            seed=seed,
            max_duration=duration,
            cfg_scale=GUIDANCE,
            top_k=TOP_K,
        )
        workflow["7"]["inputs"].update(
            seed=(seed + 1) % (MAX_SEED + 1),
            steps=steps,
            cfg=GUIDANCE,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
        )
        workflow["8"] = (
            {
                "class_type": "VAEDecodeAudioTiled",
                "inputs": {"samples": ["7", 0], "vae": ["3", 0], "tile_size": 1536, "overlap": 64},
            }
            if tiled
            else {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
        )
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        workflow["9"]["inputs"].update(
            filename_prefix=f"minimax-jam/music3-{stamp}-s{seed}",
            format="flac",
        )
        workflow["9"]["inputs"].pop("format.quality", None)
        return workflow

    @staticmethod
    def _history_error(entry: dict[str, Any]) -> str:
        for message in reversed(entry.get("status", {}).get("messages", [])):
            if isinstance(message, list) and len(message) > 1 and message[0] == "execution_error":
                detail = message[1] if isinstance(message[1], dict) else {}
                return str(detail.get("exception_message") or detail.get("exception_type") or "Generation failed")
        return "Generation failed"

    @staticmethod
    def _first_audio(entry: dict[str, Any]) -> dict[str, Any] | None:
        for output in entry.get("outputs", {}).values():
            audio = output.get("audio", []) if isinstance(output, dict) else []
            if isinstance(audio, list) and audio and isinstance(audio[0], dict):
                return audio[0]
        return None

    def _generate_comfy(self, caption: str, lyrics: str, duration: float, seed: int, steps: int, tiled: bool) -> bytes:
        client_id = f"minimax-jam-{uuid.uuid4().hex}"
        body = {
            "prompt": self._workflow(caption, lyrics, duration, seed, steps, tiled),
            "client_id": client_id,
        }
        submitted = self._json("POST", "/prompt", timeout=60, json=body)
        prompt_id = str(submitted.get("prompt_id") or "") if isinstance(submitted, dict) else ""
        if not prompt_id:
            raise RuntimeError("ComfyUI did not return a prompt ID")

        deadline = time.monotonic() + 1800
        missing_polls = 0
        while time.monotonic() < deadline:
            history = self._json("GET", f"/history/{prompt_id}", timeout=30)
            entry = history.get(prompt_id) if isinstance(history, dict) else None
            if isinstance(entry, dict):
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(self._history_error(entry))
                if status.get("completed"):
                    audio = self._first_audio(entry)
                    if not audio:
                        raise RuntimeError("MiniMax completed without audio output")
                    params = urlencode(
                        {
                            "filename": audio.get("filename", ""),
                            "subfolder": audio.get("subfolder", ""),
                            "type": audio.get("type", "output"),
                        }
                    )
                    response = self.session.get(f"{self.engine_url}/view?{params}", timeout=120)
                    if response.status_code >= 400 or not response.content:
                        raise RuntimeError("MiniMax generated audio but it could not be downloaded")
                    return _to_wav(response.content)

            queue = self._json("GET", "/queue", timeout=30)
            running = any(str(item[1]) == prompt_id for item in queue.get("queue_running", []) if len(item) > 1)
            pending = any(str(item[1]) == prompt_id for item in queue.get("queue_pending", []) if len(item) > 1)
            missing_polls = 0 if running or pending else missing_polls + 1
            if missing_polls > 60:
                raise RuntimeError("ComfyUI lost the queued MiniMax generation")
            time.sleep(0.5)
        raise RuntimeError("MiniMax generation timed out")

    def _generate_audiocpp(self, caption: str, lyrics: str, duration: float, seed: int, steps: int) -> bytes:
        body = {
            "model": "minimax-music3",
            "request": {
                "text": caption,
                "lyrics": lyrics,
                "duration_seconds": duration,
                "options": {"num_inference_steps": steps, "seed": seed},
            },
        }
        result = self._json("POST", "/v1/tasks/run", timeout=1800, json=body)
        encoded = result.get("audio") if isinstance(result, dict) else None
        if not isinstance(encoded, str):
            raise RuntimeError("audio.cpp returned no MiniMax WAV audio")
        try:
            return _to_wav(base64.b64decode(encoded, validate=True))
        except ValueError as exc:
            raise RuntimeError("audio.cpp returned invalid MiniMax audio data") from exc

    def generate(
        self,
        caption: str,
        lyrics: str,
        duration: float,
        seed: int | float | str | None,
        mode: str = "auto",
        steps_override: int | None = None,
    ) -> GenerationResult:
        caption = (caption or "").strip()
        lyrics = (lyrics or "").strip()
        if not caption:
            raise ValueError("A MiniMax music description is required")
        if not lyrics:
            raise ValueError("Lyrics are required; use [instrumental] for an instrumental song")
        duration = max(10.0, min(180.0, float(duration)))
        resolved_seed = _resolve_seed(seed)
        resolved_mode, steps, tiled = self._resolve_mode(mode, duration)
        if steps_override is not None:
            steps = max(4, min(60, int(steps_override)))

        print(
            "[minimax] "
            f"engine={self.engine_type} mode={resolved_mode} duration={duration} "
            f"seed={resolved_seed} texture_seed={(resolved_seed + 1) % (MAX_SEED + 1)} "
            f"steps={steps} guidance={GUIDANCE} tiled_decode={tiled}"
        )
        with self.lock:
            if self.engine_type == "audiocpp":
                wav_bytes = self._generate_audiocpp(caption, lyrics, duration, resolved_seed, steps)
            else:
                wav_bytes = self._generate_comfy(caption, lyrics, duration, resolved_seed, steps, tiled)
        if len(wav_bytes) <= 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            raise RuntimeError("MiniMax returned an empty or invalid WAV file")
        actual_duration = _wav_duration(wav_bytes) or duration
        return GenerationResult(
            wav_bytes=wav_bytes,
            seed=resolved_seed,
            duration=actual_duration,
            mode=resolved_mode,
            steps=steps,
            tiled_decode=tiled,
        )
