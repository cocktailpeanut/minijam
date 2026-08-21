from __future__ import annotations

import ast
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_composer import (
    BLUEPRINT_SCHEMA,
    COMPOSER_PROFILES,
    SONG_SCHEMA,
    SYSTEM_PROMPT,
    LocalComposer,
    _assemble_blueprint_lyrics,
    _default_blueprint,
    _fallback_caption,
    _fallback_lyrics,
    _has_meaningful_lyrics,
    _lyric_plan,
    _lyrics_repair_schema,
    _normalize_blueprint,
    _normalize_lyrics,
    _populated_section_count,
    _requested_bpm,
    _section_line_counts,
    _section_sequence,
    _target_bars,
)
from minimax_backend import (
    GENERATION_TIMEOUT_SECONDS,
    MAX_DURATION_SECONDS,
    MiniMaxBackend,
)


HERE = Path(__file__).resolve().parent


def _lyrics_for_blueprint(blueprint: dict, line: str = "one two three four five six seven eight") -> str:
    chunks = []
    for section in blueprint["sections"]:
        chunks.append(f"[{section['tag']}]")
        chunks.extend([line] * section["target_lyric_lines"])
    return "\n".join(chunks)


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
        self.assertIn("Writer: Qwen 3.5 0.8B Q4_K_M (Tiny)", frontend)
        self.assertIn("Writer: Qwen 3.5 2B Q4_K_M (Balanced)", frontend)
        self.assertIn("Writer: Qwen 3.5 9B Q4_K_M (Quality)", frontend)
        self.assertIn("Auto (Qwen 3.5 9B for songs over 3 minutes)", frontend)
        self.assertIn('<span class="setting-label">Writer</span>', frontend)
        self.assertIn('<span class="setting-label">Music</span>', frontend)
        self.assertIn("setting.steps", frontend)
        self.assertIn("tiled decode", frontend)

    def test_frontend_consumes_streamed_generation_progress(self):
        frontend = (HERE / "index.html").read_text(encoding="utf-8")
        self.assertIn('result.type === "progress"', frontend)
        self.assertIn("setProgress(result.message, result.progress, result.elapsed)", frontend)
        self.assertIn('fill.classList.add("indeterminate")', frontend)

    def test_ui_and_api_expose_five_minute_generation(self):
        frontend = (HERE / "index.html").read_text(encoding="utf-8")
        app_source = (HERE / "app.py").read_text(encoding="utf-8")
        self.assertIn('<option value="300">5 min</option>', frontend)
        self.assertIn('"max_duration": MAX_DURATION_SECONDS', app_source)
        self.assertIn("time_limit=GENERATION_TIMEOUT_SECONDS", app_source)
        self.assertEqual(MAX_DURATION_SECONDS, 300.0)
        self.assertEqual(GENERATION_TIMEOUT_SECONDS, 3600)


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

    def test_generation_clamps_to_five_minutes(self):
        backend = self.backend()
        backend.engine_info = lambda **_: {"total_vram": 20 * 1024 ** 3}
        fake_wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 33)
        backend._generate_comfy = MagicMock(return_value=fake_wav)

        result = backend.generate("caption", "[instrumental]", 999, 7, mode="quality")

        self.assertEqual(backend._generate_comfy.call_args.args[2], MAX_DURATION_SECONDS)
        self.assertTrue(backend._generate_comfy.call_args.args[5])
        self.assertEqual(result.duration, MAX_DURATION_SECONDS)

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

    def test_comfy_progress_state_reports_sampler_progress(self):
        backend = self.backend("comfyui")
        websocket = MagicMock()
        def progress_event(value):
            return json.dumps(
                {
                    "type": "progress_state",
                    "data": {
                        "prompt_id": "prompt-1",
                        "nodes": {
                            "7": {
                                "value": value,
                                "max": 20,
                                "state": "running",
                                "node_id": "7",
                                "prompt_id": "prompt-1",
                            }
                        },
                    },
                }
            )

        websocket.recv.side_effect = [progress_event(5), progress_event(6), TimeoutError(), TimeoutError()]

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
        self.assertTrue(any("Rendering the audio… 6 / 20" == message for message, _ in events))
        self.assertTrue(any(progress == 0.381 for _, progress in events))
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

    def test_all_writer_options_use_qwen_35(self):
        self.assertEqual(
            [COMPOSER_PROFILES[key].label for key in ("tiny", "balanced", "quality")],
            [
                "Qwen 3.5 0.8B Q4_K_M",
                "Qwen 3.5 2B Q4_K_M",
                "Qwen 3.5 9B Q4_K_M",
            ],
        )

    def test_long_song_auto_uses_quality_but_explicit_option_wins(self):
        composer = LocalComposer(HERE / "composer_models")
        with patch.dict(os.environ, {"MINIMAX_COMPOSER_PROFILE": ""}, clear=False):
            self.assertEqual(composer.resolve_profile("auto", audio_duration=300).key, "quality")
            self.assertEqual(composer.resolve_profile("tiny", audio_duration=300).key, "tiny")
            self.assertEqual(composer.resolve_profile("balanced", audio_duration=300).key, "balanced")

    def test_python_writer_uses_schema_constrained_completion(self):
        llm = MagicMock()
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "{}"}}]
        }
        with tempfile.TemporaryDirectory() as directory:
            composer = LocalComposer(directory)
            result = composer._run_completion(
                llm,
                COMPOSER_PROFILES["quality"],
                "prompt",
            )
        self.assertEqual(result, "{}")
        kwargs = llm.create_chat_completion.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["schema"], SONG_SCHEMA)
        self.assertEqual(kwargs["max_tokens"], COMPOSER_PROFILES["quality"].max_tokens)

    def test_section_tags_are_normalized_for_minimax(self):
        lyrics = _normalize_lyrics("[Verse 1] first line\n[Hook] chorus line", instrumental=False)
        self.assertEqual(lyrics, "[verse]\nfirst line\n[chorus]\nchorus line")

    def test_instrumental_and_solo_production_directions_are_removed_from_lyrics(self):
        lyrics = _normalize_lyrics(
            "[verse]\nSing this\n[instrumental]\nGuitar swells here\n[solo]\nPlay a violin solo\n[outro]\nGoodnight",
            instrumental=False,
        )
        self.assertEqual(lyrics, "[verse]\nSing this\n[instrumental]\n[solo]\n[outro]\nGoodnight")

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

    def test_short_lyric_plans_follow_the_original_space_structures(self):
        self.assertEqual(_lyric_plan(30)["sections"], ("Verse", "Chorus"))
        self.assertEqual(
            _lyric_plan(60)["sections"],
            ("Verse", "Pre-Chorus", "Chorus", "Verse", "Chorus"),
        )
        self.assertEqual(
            _lyric_plan(120)["sections"],
            ("Verse", "Pre-Chorus", "Chorus", "Verse", "Chorus", "Bridge", "Chorus", "Outro"),
        )

    def test_five_minute_plan_uses_blueprint_sections(self):
        five_minute_plan = _lyric_plan(300)
        self.assertEqual(five_minute_plan["target_vocal_sections"], 12)
        self.assertIn("Instrumental", five_minute_plan["sections"])
        self.assertEqual(five_minute_plan["sections"][-1], "Outro")

    def test_five_minute_plan_rejects_an_overstuffed_lyric_draft(self):
        line = "one two three four five six seven eight nine ten"
        lyrics = "[verse]\n" + "\n".join([line] * 51) + "\n[outro]\nThe final light goes home"
        self.assertFalse(_has_meaningful_lyrics(lyrics, 300))

    def test_blueprint_bar_math_and_validation_fill_the_requested_timeline(self):
        blueprint = _default_blueprint(300, bpm=96)
        normalized = _normalize_blueprint(blueprint, 300)
        total_bars = sum(section["approximate_bars"] for section in normalized["sections"])
        self.assertEqual(total_bars, round(_target_bars(300, 96, "4/4")))
        self.assertEqual(normalized["sections"][-1]["tag"], "outro")

    def test_explicit_bpm_is_preserved_for_blueprint_fallback(self):
        self.assertEqual(_requested_bpm("128 BPM Japanese city pop"), 128)
        self.assertIsNone(_requested_bpm("Japanese city pop"))

    def test_blueprint_validation_scales_qwen_proportions_to_exact_bar_math(self):
        blueprint = _default_blueprint(300, bpm=96)
        for section in blueprint["sections"]:
            section["approximate_bars"] = max(2, section["approximate_bars"] // 2)
        normalized = _normalize_blueprint(blueprint, 300)
        self.assertEqual(
            sum(section["approximate_bars"] for section in normalized["sections"]),
            120,
        )

    def test_five_minute_blueprint_enforces_its_per_section_line_targets(self):
        blueprint = _default_blueprint(300, bpm=96)
        lyrics = _lyrics_for_blueprint(blueprint)
        self.assertEqual(
            _section_line_counts(lyrics),
            tuple(section["target_lyric_lines"] for section in blueprint["sections"]),
        )
        self.assertTrue(_has_meaningful_lyrics(lyrics, 300, blueprint, "en"))

    def test_lyrics_repair_schema_and_assembly_enforce_each_planned_section(self):
        blueprint = _default_blueprint(300, bpm=96)
        schema = _lyrics_repair_schema(blueprint)
        payload = {}
        for index, section in enumerate(blueprint["sections"], 1):
            target = section["target_lyric_lines"]
            if target <= 0:
                continue
            key = f"section_{index:02d}"
            self.assertEqual(schema["properties"][key]["minItems"], target)
            self.assertEqual(schema["properties"][key]["maxItems"], target)
            payload[key] = [f"lyric line {line}" for line in range(target)]
        lyrics = _assemble_blueprint_lyrics(payload, blueprint)
        self.assertTrue(_has_meaningful_lyrics(lyrics, 300, blueprint, "en"))

    def test_five_minute_blueprint_rejects_underfilled_sections_in_any_language(self):
        blueprint = _default_blueprint(300, bpm=128)
        chunks = []
        for section in blueprint["sections"]:
            chunks.append(f"[{section['tag']}]")
            if section["target_lyric_lines"]:
                chunks.extend(["画面の向こうへ進む"] * 2)
        lyrics = "\n".join(chunks)
        self.assertFalse(_has_meaningful_lyrics(lyrics, 300, blueprint, "ja"))

    def test_blueprint_lyrics_must_follow_the_exact_section_timeline(self):
        blueprint = _default_blueprint(300, bpm=96)
        lyrics = _lyrics_for_blueprint(blueprint)
        expected = tuple(section["tag"] for section in blueprint["sections"])
        self.assertEqual(_section_sequence(lyrics), expected)
        self.assertTrue(_has_meaningful_lyrics(lyrics, 300, blueprint, "en"))
        self.assertFalse(_has_meaningful_lyrics(lyrics.replace("[bridge]", "[verse]", 1), 300, blueprint, "en"))

    def test_blueprint_allows_a_section_whose_vocal_plan_is_wordless(self):
        blueprint = _default_blueprint(300, bpm=96)
        blueprint["sections"][0]["vocal_plan"] = "wordless opening with no lead vocal"
        blueprint["sections"][0]["target_lyric_lines"] = 0
        lyrics = _lyrics_for_blueprint(blueprint)
        self.assertTrue(_has_meaningful_lyrics(lyrics, 300, blueprint, "en"))

    def test_long_song_composition_runs_blueprint_then_song_pass(self):
        blueprint = _default_blueprint(300, bpm=96)
        song = {
            "title": "Night Signal",
            "tags": ["synth-pop", "nocturnal", "melodic"],
            "bpm": 96,
            "language": "en",
            "lyrics": _lyrics_for_blueprint(blueprint),
            "global_metadata": "Basic Attributes: bpm is 96. key is C, and scale is minor. Synth-pop.",
            "vocal_details": "Vocal Gender & Timbre: Singer A (Female), warm alto.",
            "arrangement": "Instrument Lifecycle Description (Primary/Secondary Layering): Primary: synths.",
        }
        composer = LocalComposer(HERE / "composer_models")
        llm = MagicMock()
        with (
            patch.object(composer, "_ensure_model", return_value=HERE / "model.gguf"),
            patch.object(composer, "_load_llm", return_value=llm),
            patch.object(
                composer,
                "_run_completion",
                side_effect=[json.dumps(blueprint), json.dumps(song)],
            ) as completion,
        ):
            result = composer.compose("synth pop about finding home", 300, profile="quality")

        self.assertEqual(completion.call_count, 2)
        llm.close.assert_called_once()
        self.assertIs(completion.call_args_list[0].kwargs["schema"], BLUEPRINT_SCHEMA)
        self.assertEqual(completion.call_args_list[1].kwargs["max_tokens"], 2600)
        self.assertEqual(result["bpm"], 96)
        self.assertIn("Planned Timeline at 96 BPM", result["arrangement"])


if __name__ == "__main__":
    unittest.main()
