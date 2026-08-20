from __future__ import annotations

import base64
import gc
import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

for name in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(name, None)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from gradio import Server

from local_composer import LocalComposer, _normalize_lyrics
from minimax_backend import MiniMaxBackend


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SONGS_DIR = DATA_DIR / "songs"
COMPOSER_MODELS_DIR = BASE_DIR / "composer_models"
MAX_DURATION = 180.0

SONGS_DIR.mkdir(parents=True, exist_ok=True)
COMPOSER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

composer = LocalComposer(COMPOSER_MODELS_DIR)
backend = MiniMaxBackend(BASE_DIR / "minimax_workflow.json")
engine_address = backend.engine_url.split("://", 1)[-1]
print(f"[startup] MiniMax engine: {backend.engine_type} at {engine_address}")


def _log_block(label: str, text: str) -> None:
    print(f"[{label}] ---")
    cleaned = (text or "").rstrip()
    print(cleaned if cleaned else "<empty>")
    print(f"[/{label}] ---")


def _clamp_duration(value: float | int | str | None) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = 60.0
    return max(10.0, min(MAX_DURATION, duration))


def _composer_profile_for_hardware(requested: str, generation_mode: str) -> str:
    if (requested or "auto").strip().lower() != "auto":
        return requested
    if (generation_mode or "auto").strip().lower() == "low-vram":
        return "tiny"
    try:
        total_vram = backend.engine_info().get("total_vram")
        if total_vram is not None and float(total_vram) < 12 * 1024 ** 3:
            return "tiny"
    except Exception:
        pass
    return "auto"


def _song_public_url(song_id: str, filename: str) -> str:
    return f"/media/songs/{song_id}/{filename}"


def _decorate_song(meta: dict) -> dict:
    entry = dict(meta)
    audio_file = entry.get("audio_file")
    if audio_file:
        entry["audio_url"] = _song_public_url(entry["id"], audio_file)
    return entry


def _load_feed_from_disk() -> list[dict]:
    songs: list[dict] = []
    if not SONGS_DIR.exists():
        return songs
    for song_dir in SONGS_DIR.iterdir():
        meta_path = song_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            songs.append(_decorate_song(json.loads(meta_path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    songs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    print(f"[feed] Loaded {len(songs)} saved songs")
    return songs


def _save_song(
    wav_bytes: bytes,
    *,
    description: str,
    composed: dict,
    duration: float,
    seed: int,
    generation_mode: str,
) -> dict:
    song_id = uuid.uuid4().hex[:12]
    song_dir = SONGS_DIR / song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    audio_file = f"{song_id}.wav"
    (song_dir / audio_file).write_bytes(wav_bytes)
    meta = {
        "id": song_id,
        "title": composed["title"],
        "description": description,
        "tags": composed["tags"],
        "lyrics": composed["lyrics"],
        "caption": composed["caption"],
        "bpm": composed["bpm"],
        "language": composed["language"],
        "duration": duration,
        "seed": seed,
        "generation_mode": generation_mode,
        "audio_file": audio_file,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (song_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    entry = _decorate_song(meta)
    _feed_songs.insert(0, entry)
    return entry


_feed_songs = _load_feed_from_disk()
app = Server(title="MiniJam")


@app.api(name="create", concurrency_limit=1, time_limit=1200)
def create(
    description: str,
    audio_duration: float = 60.0,
    seed: int = -1,
    community: bool = False,
    composer_profile: str = "auto",
    generation_mode: str = "auto",
    instrumental: bool = False,
) -> str:
    started_at = time.perf_counter()
    try:
        description = (description or "").strip()
        if not description:
            raise ValueError("Describe the song you want first")
        duration = _clamp_duration(audio_duration)
        print(
            "[create] "
            f"duration={duration} seed={seed} save={community} "
            f"composer_profile={composer_profile} generation_mode={generation_mode} instrumental={instrumental}"
        )
        _log_block("create.description", description)

        # On RAM-constrained machines, avoid keeping staged MiniMax weights resident
        # while llama.cpp loads the local song writer.
        backend.unload_for_composer(generation_mode)
        resolved_composer_profile = _composer_profile_for_hardware(composer_profile, generation_mode)
        compose_started = time.perf_counter()
        composed = composer.compose(
            description=description,
            audio_duration=duration,
            profile=resolved_composer_profile,
            instrumental=instrumental,
        )
        compose_elapsed = time.perf_counter() - compose_started
        gc.collect()
        _log_block("create.minimax_caption", composed["caption"])
        _log_block("create.minimax_lyrics", composed["lyrics"])

        generation_started = time.perf_counter()
        generated = backend.generate(
            caption=composed["caption"],
            lyrics=composed["lyrics"],
            duration=duration,
            seed=seed,
            mode=generation_mode,
        )
        generation_elapsed = time.perf_counter() - generation_started
        encoded = base64.b64encode(generated.wav_bytes).decode()
        result = {
            "audio": f"data:audio/wav;base64,{encoded}",
            "title": composed["title"],
            "tags": composed["tags"],
            "lyrics": composed["lyrics"],
            "caption": composed["caption"],
            "bpm": composed["bpm"],
            "language": composed["language"],
            "composer_profile": composed["composer_profile"],
            "composer_model": composed["composer_model"],
            "generation_mode": generated.mode,
            "seed": generated.seed,
            "duration": round(generated.duration, 1),
            "steps": generated.steps,
            "tiled_decode": generated.tiled_decode,
        }
        if community:
            entry = _save_song(
                generated.wav_bytes,
                description=description,
                composed=composed,
                duration=generated.duration,
                seed=generated.seed,
                generation_mode=generated.mode,
            )
            result["community_url"] = entry["audio_url"]

        total_elapsed = time.perf_counter() - started_at
        print(
            "[create timing] "
            f"compose={compose_elapsed:.2f}s generate={generation_elapsed:.2f}s total={total_elapsed:.2f}s"
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        print(f"[create ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
    finally:
        gc.collect()


@app.api(name="generate", concurrency_limit=1, time_limit=1200)
def generate(
    prompt: str,
    lyrics: str,
    audio_duration: float = 60.0,
    steps: int = 30,
    seed: int = -1,
    generation_mode: str = "auto",
) -> str:
    try:
        normalized_lyrics = _normalize_lyrics(lyrics, instrumental=False)
        generated = backend.generate(
            caption=prompt,
            lyrics=normalized_lyrics,
            duration=_clamp_duration(audio_duration),
            seed=seed,
            mode=generation_mode,
            steps_override=int(steps),
        )
        encoded = base64.b64encode(generated.wav_bytes).decode()
        return f"data:audio/wav;base64,{encoded}"
    except Exception as exc:
        print(f"[generate ERROR] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise


@app.api(name="community", concurrency_limit=4)
def community() -> str:
    return json.dumps(_feed_songs[:50], ensure_ascii=False)


@app.api(name="config", concurrency_limit=8)
def config(audio_duration: float = 60.0) -> str:
    info = backend.engine_info()
    duration = _clamp_duration(audio_duration)
    generation_mode_settings = {}
    for requested_mode in ("auto", "low-vram", "quality"):
        resolved_mode, steps, tiled_decode = backend._resolve_mode(requested_mode, duration)
        generation_mode_settings[requested_mode] = {
            "resolved_mode": resolved_mode,
            "steps": steps,
            "tiled_decode": tiled_decode,
        }
    return json.dumps(
        {
            "active_generation_mode": "auto",
            "default_generation_mode": "auto",
            "available_generation_modes": ["low-vram", "quality"],
            "generation_mode_settings": generation_mode_settings,
            "engine": info.get("engine"),
            "engine_label": info.get("label"),
            "max_duration": MAX_DURATION,
        }
    )


@app.get("/media/songs/{song_id}/{filename}")
async def media(song_id: str, filename: str):
    songs_root = SONGS_DIR.resolve()
    song_dir = (SONGS_DIR / song_id).resolve()
    target = (song_dir / filename).resolve()
    if songs_root not in song_dir.parents or not song_dir.is_dir() or song_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


demo = app


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False)
