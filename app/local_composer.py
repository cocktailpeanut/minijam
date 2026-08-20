from __future__ import annotations

import gc
import json
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download


SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
        },
        "bpm": {"type": "integer"},
        "language": {"type": "string"},
        "lyrics": {"type": "string"},
        "global_metadata": {"type": "string"},
        "vocal_details": {"type": "string"},
        "arrangement": {"type": "string"},
    },
    "required": [
        "title",
        "tags",
        "bpm",
        "language",
        "lyrics",
        "global_metadata",
        "vocal_details",
        "arrangement",
    ],
}

CAPTION_CONTRACT = """The three caption fields must follow the labeled style MiniMax Music 3 was trained on. Be concrete and musical; describe an energy arc and instrument lifecycles, never a static equipment list or decorative adjectives. Never contradict an explicit user constraint. Do not quote or paraphrase lyric lines inside the caption. Aim for roughly 250-400 words across all three fields.

`global_metadata`: one paragraph, in order: "Basic Attributes: bpm is <number>. key is <letter>, and scale is <major|minor>. <Genre / Subgenre>." then "Global Emotional Progression: <how emotion evolves from opening through the final section>." then "Application Scenarios & Imagery: <two or three vivid listening scenarios>." then "Sonics & Production Profile: <soundstage, frequency balance, dynamics, production character>."

`vocal_details`: one paragraph: "Vocal Gender & Timbre: Singer A (<Male|Female>), <timbre and register>." then "Vocal Style: <delivery and how it shifts per section>." then "Harmony/Backing Vocals: <where harmonies or doubles appear>." then "Vocal FX: <restrained reverb, delay and compression>." For an instrumental write "Instrumental, no vocals." and name the lead melodic instrument or texture.

`arrangement`: one paragraph: "Instrument Lifecycle Description (Primary/Secondary Layering): Primary: <core instruments present start to finish and their role>. Secondary: <instruments that enter, exit or intensify, and in which sections>." then "Groove & Foundation Progression: <how drums, bass and groove develop>." then "Embellishments, Textures & Spatial FX: <fills, transitions, stereo and space treatment>." Align the changes with the lyric section tags."""

LYRICS_RULES = """Use only these section tags, alone on their own line: [intro] [verse] [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] [solo] [outro]. Never put words on the same line as a tag. Roughly 12-16 sung words per 10 seconds is an upper guide. Musical directions, sound effects and stage directions never belong in the lyrics. For an instrumental, use [instrumental] with no sung words. Write lyrics in the requested language, defaulting to English."""

SYSTEM_PROMPT = f"""You write inputs for MiniMax Music 3, a lyrics-and-description music generation model.

Reply with exactly one JSON object and nothing else.

Rules:
- `title` must be a short, catchy song title.
- `tags` must be an array of 3 to 6 concise feed-card style tags.
- `bpm` must be a plausible tempo integer.
- `language` must be one of: en, zh, ja, ko, instrumental, unknown.
- `lyrics` must be a single string. {LYRICS_RULES}
- If the request is instrumental, set `language` to `instrumental`; vocal_details must never describe a singer.
- Unless explicitly instrumental, the song has a singer and vocal_details must describe that singer.
- Match the lyric length and number of sections to the requested duration and section plan.
- For non-instrumental songs, every section marker must be followed by actual sung lyric lines.
- Never return empty sections or placeholder markers such as [END], [LYRICS], [LYRITIC], or repeated labels without lyrics.
- Produce `global_metadata`, `vocal_details`, and `arrangement`. {CAPTION_CONTRACT}
- Never wrap the JSON in markdown fences.
"""


@dataclass(frozen=True)
class ComposerProfile:
    key: str
    repo_id: str
    filename: str
    label: str
    n_ctx: int
    max_tokens: int


COMPOSER_PROFILES = {
    "tiny": ComposerProfile(
        key="tiny",
        repo_id="ggml-org/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q4_0.gguf",
        label="Qwen3 0.6B Q4_0",
        n_ctx=4096,
        max_tokens=1600,
    ),
    "balanced": ComposerProfile(
        key="balanced",
        repo_id="Qwen/Qwen3-1.7B-GGUF",
        filename="Qwen3-1.7B-Q8_0.gguf",
        label="Qwen3 1.7B Q8_0",
        n_ctx=6144,
        max_tokens=2200,
    ),
    "quality": ComposerProfile(
        key="quality",
        repo_id="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        label="Qwen3 4B Q4_K_M",
        n_ctx=8192,
        max_tokens=2600,
    ),
}

STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "at",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "lyrics",
    "music",
    "of",
    "on",
    "song",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


def _system_memory_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return (page_size * page_count) / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        return None


def _gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except Exception:
        return None
    return None


def _is_apple_mps() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import torch

        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        # The lightweight app environment intentionally does not install PyTorch.
        # Apple Silicon still uses the CPU-first composer plus the Metal music engine.
        return True


def _strip_wrappers(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_wrappers(raw)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("model did not return JSON")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("model returned non-object JSON")
    return payload


def _guess_title(description: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", description)
    if not words:
        return "Untitled"
    return " ".join(words[:5]).title()[:48].strip() or "Untitled"


def _lyric_plan(audio_duration: float) -> dict[str, Any]:
    if audio_duration >= 105:
        return {
            "structure": "[Verse], [Chorus], [Verse], [Chorus], [Bridge], [Chorus], optional [Outro]",
            "line_range": "12 to 20",
            "min_lines": 10,
            "min_words": 52,
            "sections": ("Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus", "Outro"),
        }
    if audio_duration >= 75:
        return {
            "structure": "[Verse], [Chorus], [Verse], [Chorus], optional [Bridge]",
            "line_range": "8 to 14",
            "min_lines": 8,
            "min_words": 36,
            "sections": ("Verse", "Chorus", "Verse", "Chorus", "Bridge"),
        }
    if audio_duration >= 45:
        return {
            "structure": "[Verse], [Chorus], [Verse], [Chorus]",
            "line_range": "6 to 10",
            "min_lines": 6,
            "min_words": 24,
            "sections": ("Verse", "Chorus", "Verse", "Chorus"),
        }
    return {
        "structure": "[Verse], [Chorus], optional [Bridge]",
        "line_range": "4 to 8",
        "min_lines": 4,
        "min_words": 16,
        "sections": ("Verse", "Chorus"),
    }


def _subject_terms(description: str) -> list[str]:
    source = description.strip().lower()
    if " about " in source:
        source = source.split(" about ", 1)[1]
    words = re.findall(r"[A-Za-z0-9']+", source)
    terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in STOP_WORDS or word.isdigit():
            continue
        if word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) == 4:
            break
    return terms


def _fallback_lines(section: str, section_index: int, hook: str, theme: str, accent: str) -> list[str]:
    if section == "Verse":
        variants = [
            [
                f"{hook} in the air while the {theme} starts to rise",
                "We lean into the feeling and let it color the night",
            ],
            [
                f"Every little spark of {accent} keeps the whole room bright",
                "We sing it like a secret that finally found the light",
            ],
            [
                "Another wave of heat makes the windows start to shake",
                f"We laugh into the echo of every move we make",
            ],
        ]
        return variants[min(section_index, len(variants) - 1)]

    if section == "Chorus":
        variants = [
            [
                f"{hook}, keep the fire moving through the night",
                f"{theme.title()}, in the rhythm everything feels right",
            ],
            [
                f"{hook}, turn the hunger into something we can sing",
                "Hold the heat a little higher, let the whole room ring",
            ],
        ]
        return variants[min(section_index, len(variants) - 1)]

    if section == "Bridge":
        return [
            f"We ride the taste of {theme} like a midnight wave",
            "Let the beat go softer just before it breaks",
        ]

    return [
        f"{hook} on our lips as the final lights grow thin",
        "We carry that flavor with us when the next song begins",
    ]


def _fallback_lyrics(title: str, description: str, audio_duration: float, instrumental: bool) -> str:
    if instrumental:
        return "[Instrumental]"
    hook = title or "Midnight Echo"
    terms = _subject_terms(description)
    theme = " ".join(terms[:2]).strip() or "midnight heat"
    accent = terms[2] if len(terms) >= 3 else (terms[0] if terms else "rhythm")
    plan = _lyric_plan(audio_duration)
    section_counts: dict[str, int] = {}
    chunks: list[str] = []
    for section in plan["sections"]:
        count = section_counts.get(section, 0)
        section_counts[section] = count + 1
        lines = _fallback_lines(section, count, hook, theme, accent)
        chunks.append(f"[{section}]\n" + "\n".join(lines))
    return "\n\n".join(chunks)


def _normalize_tags(tags: Any, description: str) -> list[str]:
    if isinstance(tags, str):
        candidates = re.split(r"[,/;|]", tags)
    elif isinstance(tags, list):
        candidates = tags
    else:
        candidates = []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        tag = str(item).strip().lower()
        if not tag:
            continue
        if len(tag) > 28:
            tag = tag[:28].strip()
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
        if len(normalized) == 6:
            break

    if len(normalized) >= 3:
        return normalized

    fallback = [
        "melodic",
        "studio production",
        "cinematic",
        "modern",
        "emotional",
        "songwriting",
    ]
    if "lofi" in description.lower() or "lo-fi" in description.lower():
        fallback.insert(0, "lo-fi")
    elif "rock" in description.lower():
        fallback.insert(0, "rock")
    elif "rap" in description.lower() or "hip hop" in description.lower():
        fallback.insert(0, "hip-hop")
    else:
        fallback.insert(0, "pop")

    for tag in fallback:
        if tag not in seen:
            normalized.append(tag)
            seen.add(tag)
        if len(normalized) == 4:
            break
    return normalized


def _normalize_lyrics(lyrics: Any, instrumental: bool) -> str:
    if instrumental:
        return "[instrumental]"

    text = str(lyrics or "").replace("\r", "").replace("\\n", "\n").strip()
    if not text:
        return ""
    if "[" not in text:
        text = f"[verse]\n{text}"

    aliases = {
        "hook": "chorus",
        "refrain": "chorus",
        "interlude": "instrumental",
        "break": "instrumental",
        "ending": "outro",
    }
    allowed = {
        "intro",
        "verse",
        "pre-chorus",
        "chorus",
        "post-chorus",
        "bridge",
        "instrumental",
        "solo",
        "outro",
    }
    out: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", line)
        if not match:
            out.append(line.rstrip())
            continue
        raw_tag = re.sub(r"\s+\d+\s*$", "", match.group(1).strip().lower())
        tag = aliases.get(raw_tag, raw_tag)
        remainder = match.group(2).strip()
        if tag in allowed:
            out.append(f"[{tag}]")
            if remainder:
                out.append(remainder)
        elif remainder:
            out.append(remainder)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _has_meaningful_lyrics(text: str, audio_duration: float) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("[end]", "[lyrics]", "[lyritic]", "[end song]")):
        return False

    plan = _lyric_plan(audio_duration)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lyric_lines: list[str] = []
    for line in nonempty_lines:
        stripped = re.sub(r"\[[^\]]+\]", "", line).strip()
        if stripped:
            lyric_lines.append(stripped)

    if len(lyric_lines) < plan["min_lines"]:
        return False

    body = "\n".join(lyric_lines)
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", body, re.UNICODE)
    return len(words) >= plan["min_words"]


def _duration_prompt(audio_duration: float, instrumental: bool) -> str:
    if instrumental:
        return "Keep the output instrumental. Use [instrumental] section tags with no sung words."
    plan = _lyric_plan(audio_duration)
    return (
        f"Use this section plan: {plan['structure']}.\n"
        f"Write {plan['line_range']} non-empty lyric lines total.\n"
        "Every section must include actual sung lines, not empty labels.\n"
        "Do not emit placeholder tokens such as [END], [LYRICS], or [Instrumental]."
    )


def _fallback_caption(
    description: str,
    tags: list[str],
    bpm: int,
    instrumental: bool,
) -> tuple[str, str, str]:
    genre = tags[0].title() if tags else "Contemporary Pop"
    mood = tags[1] if len(tags) > 1 else "emotionally direct"
    global_metadata = (
        f"Basic Attributes: bpm is {bpm}. key is C, and scale is major. {genre}. "
        f"Global Emotional Progression: The opening establishes a {mood} atmosphere, the middle sections "
        "broaden in intensity, and the final section resolves with the strongest emotional statement. "
        f"Application Scenarios & Imagery: {description.strip()[:220] or 'A vivid late-night listening scene and an intimate live performance'}. "
        "Sonics & Production Profile: A balanced modern mix with clear transients, controlled low end, "
        "natural dynamics, and a wide but focused stereo image."
    )
    vocal_details = (
        "Instrumental, no vocals. The lead melodic role moves between the primary instrument and a supporting texture."
        if instrumental
        else "Vocal Gender & Timbre: Singer A (Female), a clear mid-register voice with a warm edge. "
        "Vocal Style: Intimate and measured in the verses, opening into a stronger sustained chorus delivery. "
        "Harmony/Backing Vocals: Restrained doubles enter in the choruses and widen in the final refrain. "
        "Vocal FX: Short plate reverb, subtle delay, and light transparent compression."
    )
    arrangement = (
        "Instrument Lifecycle Description (Primary/Secondary Layering): Primary: drums, bass, and the central harmonic instrument "
        "establish the song and remain coherent throughout. Secondary: supporting textures enter after the opening, intensify through "
        "the choruses, thin out before the bridge, and return for the final section. Groove & Foundation Progression: the rhythm starts "
        "restrained, adds weight and movement section by section, and reaches its fullest form near the ending. Embellishments, Textures "
        "& Spatial FX: concise fills connect sections while filtered transitions, stereo ambience, and a controlled reverb tail create depth."
    )
    return global_metadata, vocal_details, arrangement


def _log_block(label: str, text: str) -> None:
    print(f"[{label}] ---")
    cleaned = (text or "").rstrip()
    print(cleaned if cleaned else "<empty>")
    print(f"[/{label}] ---")


class LocalComposer:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def resolve_profile(
        self,
        requested: str | None,
        audio_duration: float = 60.0,
        instrumental: bool = False,
    ) -> ComposerProfile:
        override = os.environ.get("MINIMAX_COMPOSER_PROFILE", "").strip().lower()
        key = (override or requested or "auto").strip().lower()

        if key == "auto":
            ram_gb = _system_memory_gb()
            vram_gb = _gpu_memory_gb()
            long_vocal_request = (not instrumental) and audio_duration >= 90
            if _is_apple_mps():
                # Metal offload keeps short requests fast while unified-memory Macs can
                # use the stronger composer for long vocal structures.
                key = "quality" if long_vocal_request else "tiny"
            elif long_vocal_request:
                if (ram_gb is not None and ram_gb >= 24) or (vram_gb is not None and vram_gb >= 16):
                    key = "quality"
                else:
                    key = "balanced"
            elif (vram_gb is not None and vram_gb <= 8) or (ram_gb is not None and ram_gb < 16):
                key = "tiny"
            elif (ram_gb is not None and ram_gb >= 24) or (vram_gb is not None and vram_gb >= 16):
                key = "quality"
            else:
                key = "balanced"

        if key not in COMPOSER_PROFILES:
            key = "balanced"
        return COMPOSER_PROFILES[key]

    def compose(
        self,
        description: str,
        audio_duration: float = 60.0,
        profile: str = "auto",
        instrumental: bool = False,
        progress_callback: Callable[[str, float | None], None] | None = None,
    ) -> dict[str, Any]:
        def progress(message: str, value: float | None = None) -> None:
            if progress_callback is not None:
                progress_callback(message, value)

        compose_started_at = time.perf_counter()
        selected = self.resolve_profile(profile, audio_duration=audio_duration, instrumental=instrumental)
        print(
            "[composer] "
            f"starting profile_request={profile} "
            f"profile_resolved={selected.key} "
            f"model={selected.label} "
            f"duration={audio_duration} "
            f"instrumental={instrumental}"
        )
        progress(f"Preparing {selected.label} writer…")
        model_path = self._ensure_model(selected)
        ensure_elapsed = time.perf_counter() - compose_started_at
        print(f"[composer] model ready path={model_path} elapsed={ensure_elapsed:.2f}s")
        progress(f"Loading {selected.label} writer…")
        load_started_at = time.perf_counter()
        llm = self._load_llm(selected, model_path)
        load_elapsed = time.perf_counter() - load_started_at
        print(f"[composer] llama loaded elapsed={load_elapsed:.2f}s")

        user_prompt = (
            f"Description: {description.strip()}\n"
            f"Instrumental: {'yes' if instrumental else 'no'}\n"
            f"Target duration seconds: {int(audio_duration)}\n"
            f"{_duration_prompt(audio_duration, instrumental)}\n"
            "Write the song spec now."
        )
        _log_block("composer.prompt", user_prompt)

        try:
            generation_started_at = time.perf_counter()
            print("[composer] generating song spec...")
            progress("Writing the title, lyrics, and production plan…")
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_object",
                    "schema": SONG_SCHEMA,
                },
                temperature=0.8,
                top_p=0.92,
                repeat_penalty=1.05,
                max_tokens=selected.max_tokens,
            )
            content = response["choices"][0]["message"]["content"] or "{}"
            generation_elapsed = time.perf_counter() - generation_started_at
            print(f"[composer] completion received elapsed={generation_elapsed:.2f}s chars={len(content)}")
            progress("Checking the song structure…")
            _log_block("composer.raw_response", content)
            payload = _extract_json(content)
            print(f"[composer] parsed response keys={sorted(payload.keys())}")
        except Exception:
            payload = {}
        finally:
            closer = getattr(llm, "close", None)
            if callable(closer):
                closer()
            del llm
            gc.collect()

        title = str(payload.get("title") or _guess_title(description)).strip()[:60] or "Untitled"
        tags = _normalize_tags(payload.get("tags"), description)
        bpm = payload.get("bpm")
        try:
            bpm_value = int(bpm)
        except (TypeError, ValueError):
            bpm_value = 120
        bpm_value = min(180, max(60, bpm_value))

        language = str(payload.get("language") or ("instrumental" if instrumental else "en")).strip().lower()
        if language not in {"en", "zh", "ja", "ko", "instrumental", "unknown"}:
            language = "instrumental" if instrumental else "en"

        lyrics = _normalize_lyrics(payload.get("lyrics"), instrumental)
        used_fallback_lyrics = False
        if not instrumental and (language == "instrumental" or not _has_meaningful_lyrics(lyrics, audio_duration)):
            language = "en"
            lyrics = _fallback_lyrics(title, description, audio_duration, instrumental=False)
            lyrics = _normalize_lyrics(lyrics, instrumental=False)
            used_fallback_lyrics = True

        fallback_global, fallback_vocals, fallback_arrangement = _fallback_caption(
            description,
            tags,
            bpm_value,
            instrumental,
        )
        global_metadata = str(payload.get("global_metadata") or fallback_global).strip()
        vocal_details = str(payload.get("vocal_details") or fallback_vocals).strip()
        arrangement = str(payload.get("arrangement") or fallback_arrangement).strip()
        if instrumental and not vocal_details.lower().startswith("instrumental"):
            vocal_details = fallback_vocals
        if not instrumental and vocal_details.lower().startswith("instrumental"):
            vocal_details = fallback_vocals
        caption = "\n".join((global_metadata, vocal_details, arrangement))

        total_elapsed = time.perf_counter() - compose_started_at
        print(
            "[composer] "
            f"done profile={selected.key} "
            f"language={language} "
            f"bpm={bpm_value} "
            f"fallback_lyrics={used_fallback_lyrics} "
            f"total={total_elapsed:.2f}s"
        )
        print(f"[composer] title={title}")
        print(f"[composer] tags={', '.join(tags)}")
        _log_block("composer.final_lyrics", lyrics)
        _log_block("composer.minimax_caption", caption)

        return {
            "title": title,
            "tags": ", ".join(tags),
            "bpm": bpm_value,
            "language": language,
            "lyrics": lyrics,
            "caption": caption,
            "global_metadata": global_metadata,
            "vocal_details": vocal_details,
            "arrangement": arrangement,
            "composer_profile": selected.key,
            "composer_model": selected.label,
        }

    def _ensure_model(self, profile: ComposerProfile) -> Path:
        model_dir = self.models_dir / profile.key
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(
            repo_id=profile.repo_id,
            filename=profile.filename,
            local_dir=model_dir,
        )
        return Path(model_path)

    def _load_llm(self, profile: ComposerProfile, model_path: Path):
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed in app/env. Re-run Install or Update."
            ) from exc

        default_gpu_layers = -1 if _is_apple_mps() else 0
        configured_gpu_layers = os.environ.get("MINIMAX_COMPOSER_GPU_LAYERS", "").strip()
        gpu_layers = int(configured_gpu_layers) if configured_gpu_layers else default_gpu_layers
        gpu_layers = max(-1, gpu_layers)
        print(
            "[composer] "
            f"llama backend={'metal' if _is_apple_mps() and gpu_layers != 0 else 'cpu'} "
            f"gpu_layers={gpu_layers}"
        )
        return Llama(
            model_path=str(model_path),
            n_ctx=profile.n_ctx,
            n_batch=min(512, profile.n_ctx),
            n_gpu_layers=gpu_layers,
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
