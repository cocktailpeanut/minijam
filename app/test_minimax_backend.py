from __future__ import annotations

import ast
import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_composer import _fallback_caption, _normalize_lyrics
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


class ComposerContractTests(unittest.TestCase):
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
