# MiniJam app

Local runtime for the Pinokio launcher in the parent folder.

## Flow

1. `local_composer.py` turns a plain-language idea into a title, feed tags, tagged lyrics, and MiniMax's Global Metadata / Vocal Details / Arrangement caption.
2. `minimax_backend.py` translates each request to the ComfyUI INT8 or audio.cpp Q4 engine.
3. `app.py` returns WAV audio and optionally stores it under `data/songs/`.

The frontend remains intentionally thin. Low-memory mode, tiled decoding, Space-compatible seeds, and engine-specific request formats are handled by the backend.

Run through the parent Pinokio launcher. It supplies `MINIMAX_ENGINE_URL` and `MINIMAX_ENGINE` after starting the local engine.
