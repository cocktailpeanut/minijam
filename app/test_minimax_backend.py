from __future__ import annotations

import ast
import base64
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_composer import (
    COMPOSER_PROFILES,
    SYSTEM_PROMPT,
    LocalComposer,
    _fallback_caption,
    _normalize_lyrics,
)
from minimax_backend import MiniMaxBackend


HERE = Path(__file__).resolve().parent


class ApiContractTests(unittest.TestCase):
    def test_public_api_uses_minimax_generation_names(self):
        tree = ast.parse((HERE / "app.py").read_text(encoding="utf-8"))
        functions = {
            node.name: [argument.arg for argument in node.args.args]
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(
            functions["create"],
            [
                "description",
                "audio_duration",
                "seed",
                "community",
                "composer_profile",
                "generation_mode",
                "instrumental",
            ],
        )
        self.assertEqual(
            functions["generate"],
            ["prompt", "lyrics", "audio_duration", "steps", "seed", "generation_mode"],
        )
        self.assertEqual(functions["config"], ["audio_duration"])

    def test_frontend_names_actual_writer_models_and_music_settings(self):
        frontend = (HERE / "index.html").read_text(encoding="utf-8")
        self.assertIn("Writer: Qwen3 0.6B Q4_0 (Tiny)", frontend)
        self.assertIn("Writer: Qwen3 1.7B Q8_0 (Balanced)", frontend)
        self.assertIn("Writer: Qwen3 4B Q4_K_M (Quality)", frontend)
        self.assertIn('<span class="setting-label">Writer</span>', frontend)
        self.assertIn('<span class="setting-label">Music</span>', frontend)
        self.assertIn("setting.steps", frontend)
        self.assertIn("tiled decode", frontend)

    def test_frontend_consumes_streamed_generation_progress(self):
        frontend = (HERE / "index.html").read_text(encoding="utf-8")
        self.assertIn('result.type === "progress"', frontend)
        self.assertIn("setProgress(result.message, result.progress, result.elapsed)", frontend)
        self.assertIn('fill.classList.add("indeterminate")', frontend)


class MiniMaxBackendTests(unittest.TestCase):
    def backend(self, engine: str = "comfyui") -> MiniMaxBackend:
        with patch.dict(
            os.environ,
            {"MINIMAX_ENGINE_URL": "http://127.0.0.1:9999", "MINIMAX_ENGINE": engine},
            clear=False,
        ):
            return MiniMaxBackend(HERE / "minimax_workflow.json")

    def test_auto_uses_low_vram_profile_on_eight_gb(self):
        backend = self.backend()
        backend.engine_info = lambda **_: {"total_vram": 8 * 1024 ** 3}
        self.assertEqual(backend._resolve_mode("auto", 60), ("low-vram", 20, True))

    def test_quality_preserves_space_settings_when_memory_allows(self):
        backend = self.backend()
        backend.engine_info = lambda **_: {"total_vram": 20 * 1024 ** 3}
        self.assertEqual(backend._resolve_mode("quality", 60), ("quality", 30, False))
        self.assertEqual(backend._resolve_mode("quality", 120), ("quality", 30, True))

    def test_workflow_uses_seed_plus_one_for_texture(self):
        backend = self.backend()
        workflow = backend._workflow("caption", "[verse]\nline", 60, 77, 30, True)
        self.assertEqual(workflow["4"]["inputs"]["seed"], 77)
        self.assertEqual(workflow["7"]["inputs"]["seed"], 78)
        self.assertEqual(workflow["4"]["inputs"]["cfg_scale"], 1.7)
        self.assertEqual(workflow["4"]["inputs"]["top_k"], 50)
        self.assertEqual(workflow["7"]["inputs"]["steps"], 30)
        self.assertEqual(workflow["8"]["class_type"], "VAEDecodeAudioTiled")
        json.dumps(workflow)

    def test_audiocpp_defaults_to_fast_q4_steps(self):
        backend = self.backend("audiocpp")
        self.assertEqual(backend._resolve_mode("auto", 60), ("auto", 15, False))

    def test_audiocpp_unload_targets_the_music_model(self):
        backend = self.backend("audiocpp")
        response = MagicMock()
        backend.session.post = MagicMock(return_value=response)
        with patch("minimax_backend._system_memory_gb", return_value=16):
            backend.unload_for_composer("low-vram")
        backend.session.post.assert_called_once_with(
            "http://127.0.0.1:9999/v1/tasks/unload_models",
            timeout=60,
            json={"model_ids": ["minimax-music3"]},
        )
        response.raise_for_status.assert_called_once_with()

    def test_audiocpp_stream_reports_real_stage_progress(self):
        backend = self.backend("audiocpp")
        response = MagicMock(status_code=200)
        response.iter_lines.return_value = [
            'data: {"type":"progress","stage":"flow","step":3,"total":15}',
            "",
            'data: {"type":"result","result":{"audio":"'
            + base64.b64encode(b"audio").decode()
            + '"}}',
            "",
        ]
        backend.session.post = MagicMock(return_value=response)
        events = []
        with patch("minimax_backend._to_wav", return_value=b"decoded"):
            result = backend._generate_audiocpp(
                "caption",
                "[instrumental]",
                30,
                7,
                15,
                lambda message, progress: events.append((message, progress)),
            )
        self.assertEqual(result, b"decoded")
        self.assertTrue(any("Rendering the song" in message for message, _ in events))
        self.assertTrue(any(progress is not None for _, progress in events))
        backend.session.post.assert_called_once()
        self.assertTrue(backend.session.post.call_args.kwargs["stream"])
        self.assertTrue(backend.session.post.call_args.args[0].endswith("/v1/tasks/run-stream"))
        response.close.assert_called_once_with()

    def test_comfy_websocket_reports_sampler_progress(self):
        backend = self.backend("comfyui")
        websocket = MagicMock()
        websocket.recv.return_value = json.dumps(
            {
                "type": "progress",
                "data": {"prompt_id": "prompt-1", "value": 5, "max": 20},
            }
        )

        def fake_json(method, path, **_kwargs):
            if method == "POST" and path == "/prompt":
                return {"prompt_id": "prompt-1"}
            if method == "GET" and path == "/history/prompt-1":
                return {
                    "prompt-1": {
                        "status": {"completed": True},
                        "outputs": {
                            "9": {
                                "audio": [
                                    {"filename": "song.flac", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                }
            raise AssertionError((method, path))

        backend._json = fake_json
        backend.session.get = MagicMock(return_value=MagicMock(status_code=200, content=b"audio"))
        events = []
        with (
            patch("websockets.sync.client.connect", return_value=websocket),
            patch("minimax_backend._to_wav", return_value=b"decoded"),
        ):
            result = backend._generate_comfy(
                "caption",
                "[instrumental]",
                30,
                7,
                20,
                True,
                lambda message, progress: events.append((message, progress)),
            )
        self.assertEqual(result, b"decoded")
        self.assertTrue(any("Rendering the audio" in message for message, _ in events))
        self.assertTrue(any(progress == 0.25 for _, progress in events))
        websocket.close.assert_called_once_with()


class ComposerContractTests(unittest.TestCase):
    def test_writer_prompt_separates_style_from_song_subject(self):
        self.assertIn("musical direction from lyrical subject", SYSTEM_PROMPT)
        self.assertIn('"a <style> song about <subject>"', SYSTEM_PROMPT)
        self.assertIn("make the title and lyrics about <subject>", SYSTEM_PROMPT)

    def test_apple_auto_writer_profile_uses_available_unified_memory(self):
        composer = LocalComposer(HERE / "composer_models")
        cases = ((12, "tiny"), (20, "balanced"), (64, "quality"))
        for ram_gb, expected in cases:
            with (
                self.subTest(ram_gb=ram_gb),
                patch("local_composer._is_apple_mps", return_value=True),
                patch("local_composer._system_memory_gb", return_value=ram_gb),
                patch("local_composer._gpu_memory_gb", return_value=None),
                patch.dict(os.environ, {"MINIMAX_COMPOSER_PROFILE": ""}, clear=False),
            ):
                selected = composer.resolve_profile("auto", audio_duration=60, instrumental=False)
            self.assertEqual(selected.key, expected)

    def test_apple_silicon_writer_defaults_to_full_metal_offload(self):
        llama = MagicMock()
        composer = LocalComposer(HERE / "composer_models")
        fake_module = types.SimpleNamespace(Llama=llama)
        with (
            patch("local_composer._is_apple_mps", return_value=True),
            patch.dict(os.environ, {"MINIMAX_COMPOSER_GPU_LAYERS": ""}, clear=False),
            patch.dict(sys.modules, {"llama_cpp": fake_module}),
        ):
            composer._load_llm(COMPOSER_PROFILES["tiny"], HERE / "unused.gguf")
        self.assertEqual(llama.call_args.kwargs["n_gpu_layers"], -1)

    def test_writer_gpu_layer_override_can_force_cpu(self):
        llama = MagicMock()
        composer = LocalComposer(HERE / "composer_models")
        fake_module = types.SimpleNamespace(Llama=llama)
        with (
            patch("local_composer._is_apple_mps", return_value=True),
            patch.dict(os.environ, {"MINIMAX_COMPOSER_GPU_LAYERS": "0"}, clear=False),
            patch.dict(sys.modules, {"llama_cpp": fake_module}),
        ):
            composer._load_llm(COMPOSER_PROFILES["tiny"], HERE / "unused.gguf")
        self.assertEqual(llama.call_args.kwargs["n_gpu_layers"], 0)

    def test_section_tags_are_normalized_for_minimax(self):
        lyrics = _normalize_lyrics("[Verse 1] first line\n[Hook] chorus line", instrumental=False)
        self.assertEqual(lyrics, "[verse]\nfirst line\n[chorus]\nchorus line")

    def test_fallback_caption_has_three_minimax_parts(self):
        global_metadata, vocals, arrangement = _fallback_caption(
            "dreamy synth-pop about leaving home",
            ["synth-pop", "dreamy", "female vocals"],
            105,
            False,
        )
        self.assertIn("Basic Attributes:", global_metadata)
        self.assertIn("Vocal Gender & Timbre:", vocals)
        self.assertIn("Instrument Lifecycle Description", arrangement)


if __name__ == "__main__":
    unittest.main()
