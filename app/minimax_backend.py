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
from typing import Any, Callable, Iterator
from urllib.parse import urlencode

import requests
import soundfile as sf


MAX_SEED = 2_147_483_647
MAX_DURATION_SECONDS = 300.0
GENERATION_TIMEOUT_SECONDS = 3600
DEFAULT_STEPS = 30
LOW_VRAM_STEPS = 20
GUIDANCE = 1.7
TOP_K = 50
ProgressCallback = Callable[[str, float | None], None]


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
                "memory_mode": os.environ.get("MINIMAX_AUDIOCPP_MEMORY_MODE", "low-memory"),
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
                "memory_mode": "dynamic-vram",
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

    @staticmethod
    def _comfy_phase(node_id: str, tiled: bool) -> tuple[str, float | None]:
        phases = {
            "1": ("Loading the INT8 diffusion model…", None),
            "2": ("Loading the music encoder…", None),
            "3": ("Loading the audio decoder…", None),
            "4": ("Composing structure and performance…", None),
            "5": ("Preparing diffusion conditioning…", None),
            "6": ("Preparing the audio canvas…", None),
            "7": ("Rendering the audio…", 0.15),
            "8": ("Decoding in low-memory tiles…" if tiled else "Decoding the song…", 0.94),
            "9": ("Saving the finished song…", 0.99),
        }
        return phases.get(node_id, ("Generating with ComfyUI…", None))

    @staticmethod
    def _comfy_step_progress(
        node_id: str,
        value: float,
        total: float,
        tiled: bool,
    ) -> tuple[str, float | None]:
        labels = {
            "4": "Composing structure and performance",
            "7": "Rendering the audio",
            "8": "Decoding in low-memory tiles" if tiled else "Decoding the song",
        }
        spans = {
            "4": (0.0, 0.15),
            "7": (0.15, 0.92),
            "8": (0.92, 0.99),
        }
        ratio = max(0.0, min(1.0, value / total)) if total > 0 else None
        if node_id not in spans:
            phase, _ = MiniMaxBackend._comfy_phase(node_id, tiled)
            return phase, None
        label = labels[node_id]
        message = f"{label}… {int(value)} / {int(total)}"
        if ratio is None:
            return message, None
        start, end = spans[node_id]
        return message, start + (end - start) * ratio

    @staticmethod
    def _audiocpp_progress(stage: str, step: int, total: int) -> tuple[str, float | None]:
        labels = {
            "ar": f"Composing the song… {step} / {total}",
            "flow": f"Rendering the song… {step} / {total}",
            "vocoder": f"Decoding the audio… {step} / {total}",
        }
        spans = {
            "ar": (0.0, 0.25),
            "flow": (0.25, 0.92),
            "vocoder": (0.92, 0.99),
        }
        progress = None
        if total > 0 and stage in spans:
            start, end = spans[stage]
            progress = start + (end - start) * max(0.0, min(1.0, step / total))
        return labels.get(stage, f"{stage or 'Generating'}…"), progress

    @staticmethod
    def _sse_events(response: requests.Response) -> Iterator[dict[str, Any]]:
        data_lines: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if line:
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                continue
            if not data_lines:
                continue
            try:
                event = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                event = None
            data_lines.clear()
            if isinstance(event, dict):
                yield event
        if data_lines:
            try:
                event = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict):
                yield event

    def _generate_comfy(
        self,
        caption: str,
        lyrics: str,
        duration: float,
        seed: int,
        steps: int,
        tiled: bool,
        progress_callback: ProgressCallback | None = None,
    ) -> bytes:
        client_id = f"minimax-jam-{uuid.uuid4().hex}"
        websocket = None
        try:
            from websockets.sync.client import connect

            websocket_url = self.engine_url.replace("http://", "ws://").replace("https://", "wss://")
            websocket = connect(
                f"{websocket_url}/ws?clientId={client_id}",
                open_timeout=8,
                close_timeout=1,
                ping_interval=20,
            )
        except Exception as exc:
            print(f"[minimax] ComfyUI progress websocket unavailable; using polling: {exc}")

        def progress(message: str, value: float | None = None) -> None:
            if progress_callback is not None:
                progress_callback(message, value)

        body = {
            "prompt": self._workflow(caption, lyrics, duration, seed, steps, tiled),
            "client_id": client_id,
        }
        try:
            progress("Submitting the MiniMax workflow…")
            submitted = self._json("POST", "/prompt", timeout=60, json=body)
            prompt_id = str(submitted.get("prompt_id") or "") if isinstance(submitted, dict) else ""
            if not prompt_id:
                raise RuntimeError("ComfyUI did not return a prompt ID")

            deadline = time.monotonic() + GENERATION_TIMEOUT_SECONDS
            missing_polls = 0
            current_node = ""

            def handle_websocket_message(message: str) -> None:
                nonlocal current_node
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    return
                data = event.get("data", {}) if isinstance(event, dict) else {}
                if str(data.get("prompt_id") or "") != prompt_id:
                    return
                event_type = event.get("type")
                if event_type == "execution_start":
                    progress("Starting the MiniMax workflow…")
                elif event_type == "executing" and data.get("node") is not None:
                    current_node = str(data["node"])
                    phase, phase_progress = self._comfy_phase(current_node, tiled)
                    progress(phase, phase_progress)
                elif event_type in {"progress", "progress_state"}:
                    progress_node = current_node
                    progress_data = data
                    if event_type == "progress_state":
                        nodes = data.get("nodes", {})
                        if not isinstance(nodes, dict):
                            return
                        node_item = None
                        if current_node in nodes and isinstance(nodes[current_node], dict):
                            node_item = (current_node, nodes[current_node])
                        running_item = next(
                            (
                                (str(node_id), state)
                                for node_id, state in nodes.items()
                                if isinstance(state, dict) and state.get("state") == "running"
                            ),
                            None,
                        )
                        if running_item is not None:
                            node_item = running_item
                        if node_item is None:
                            return
                        progress_node, progress_data = node_item
                        current_node = progress_node

                    if progress_data.get("max"):
                        value = max(0.0, float(progress_data.get("value", 0)))
                        total = max(1.0, float(progress_data["max"]))
                        message_text, ratio = self._comfy_step_progress(
                            progress_node,
                            value,
                            total,
                            tiled,
                        )
                        progress(message_text, ratio)
                elif event_type == "execution_error":
                    raise RuntimeError(
                        str(data.get("exception_message") or data.get("exception_type") or "Generation failed")
                    )

            def drain_websocket(first_timeout: float = 0.1, max_messages: int = 512) -> None:
                nonlocal websocket
                if websocket is None:
                    return
                for index in range(max_messages):
                    try:
                        message = websocket.recv(timeout=first_timeout if index == 0 else 0.001)
                    except TimeoutError:
                        break
                    except Exception as exc:
                        print(f"[minimax] ComfyUI progress websocket closed; using polling: {exc}")
                        try:
                            websocket.close()
                        except Exception:
                            pass
                        websocket = None
                        break
                    if isinstance(message, str):
                        handle_websocket_message(message)

            while time.monotonic() < deadline:
                drain_websocket()

                history = self._json("GET", f"/history/{prompt_id}", timeout=30)
                entry = history.get(prompt_id) if isinstance(history, dict) else None
                if isinstance(entry, dict):
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(self._history_error(entry))
                    if status.get("completed"):
                        # ComfyUI records completion before this client necessarily
                        # consumes every progress_state message already in the socket.
                        drain_websocket(first_timeout=0.05, max_messages=1024)
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
                        progress("Loading the finished audio…", 1.0)
                        return _to_wav(response.content)

                queue = self._json("GET", "/queue", timeout=30)
                running = any(str(item[1]) == prompt_id for item in queue.get("queue_running", []) if len(item) > 1)
                pending_items = [item for item in queue.get("queue_pending", []) if len(item) > 1]
                position = next(
                    (index + 1 for index, item in enumerate(pending_items) if str(item[1]) == prompt_id),
                    None,
                )
                if running:
                    missing_polls = 0
                elif position is not None:
                    missing_polls = 0
                    progress(f"Waiting for the GPU… queue position {position}")
                else:
                    missing_polls += 1
                if missing_polls > 60:
                    raise RuntimeError("ComfyUI lost the queued MiniMax generation")
                time.sleep(0.4)
            raise RuntimeError("MiniMax generation timed out")
        finally:
            if websocket is not None:
                try:
                    websocket.close()
                except Exception:
                    pass

    def _generate_audiocpp(
        self,
        caption: str,
        lyrics: str,
        duration: float,
        seed: int,
        steps: int,
        progress_callback: ProgressCallback | None = None,
    ) -> bytes:
        body = {
            "model": "minimax-music3",
            "request": {
                "text": caption,
                "lyrics": lyrics,
                "duration_seconds": duration,
                "options": {"num_inference_steps": steps, "seed": seed},
            },
        }
        if progress_callback is not None:
            progress_callback("Preparing Metal kernels and the MiniMax model…", None)
        try:
            response = self.session.post(
                f"{self.engine_url}/v1/tasks/run-stream",
                timeout=GENERATION_TIMEOUT_SECONDS,
                json=body,
                stream=True,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"The MiniMax engine is unavailable at {self.engine_url}") from exc

        result: dict[str, Any] | None = None
        use_legacy_endpoint = response.status_code == 404
        try:
            if not use_legacy_endpoint and response.status_code >= 400:
                detail = response.text.strip()
                try:
                    parsed = response.json()
                    if isinstance(parsed, dict):
                        error = parsed.get("error")
                        if isinstance(error, dict):
                            error = error.get("message") or error.get("type")
                        detail = str(error or parsed.get("message") or detail)
                except ValueError:
                    pass
                raise RuntimeError(detail or f"audio.cpp request failed ({response.status_code})")
            if not use_legacy_endpoint:
                for event in self._sse_events(response):
                    event_type = event.get("type")
                    if event_type == "progress":
                        try:
                            step = max(0, int(event.get("step") or 0))
                            total = max(0, int(event.get("total") or 0))
                        except (TypeError, ValueError):
                            step, total = 0, 0
                        message, value = self._audiocpp_progress(str(event.get("stage") or ""), step, total)
                        if progress_callback is not None:
                            progress_callback(message, value)
                    elif event_type == "result" and isinstance(event.get("result"), dict):
                        result = event["result"]
                    elif event_type == "error":
                        error = event.get("message") or event.get("error")
                        if isinstance(error, dict):
                            error = error.get("message") or error.get("type")
                        raise RuntimeError(str(error or "audio.cpp music generation failed"))
        finally:
            response.close()

        if use_legacy_endpoint:
            print("[minimax] audio.cpp streaming endpoint unavailable; using blocking generation")
            result = self._json(
                "POST",
                "/v1/tasks/run",
                timeout=GENERATION_TIMEOUT_SECONDS,
                json=body,
            )
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
        progress_callback: ProgressCallback | None = None,
    ) -> GenerationResult:
        caption = (caption or "").strip()
        lyrics = (lyrics or "").strip()
        if not caption:
            raise ValueError("A MiniMax music description is required")
        if not lyrics:
            raise ValueError("Lyrics are required; use [instrumental] for an instrumental song")
        duration = max(10.0, min(MAX_DURATION_SECONDS, float(duration)))
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
                wav_bytes = self._generate_audiocpp(
                    caption,
                    lyrics,
                    duration,
                    resolved_seed,
                    steps,
                    progress_callback,
                )
            else:
                wav_bytes = self._generate_comfy(
                    caption,
                    lyrics,
                    duration,
                    resolved_seed,
                    steps,
                    tiled,
                    progress_callback,
                )
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
