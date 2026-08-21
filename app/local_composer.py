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
    "additionalProperties": False,
}

BLUEPRINT_SCHEMA = {
    "type": "object",
    "properties": {
        "bpm": {"type": "integer"},
        "meter": {"type": "string", "enum": ["4/4", "3/4", "6/8"]},
        "duration_use": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 8,
            "maxItems": 18,
            "items": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "enum": [
                            "intro", "verse", "pre-chorus", "chorus", "post-chorus",
                            "bridge", "instrumental", "solo", "outro",
                        ],
                    },
                    "approximate_bars": {"type": "integer", "minimum": 2, "maximum": 64},
                    "target_lyric_lines": {"type": "integer", "minimum": 0, "maximum": 64},
                    "vocal_plan": {"type": "string"},
                    "production_events": {"type": "string"},
                },
                "required": [
                    "tag", "approximate_bars", "target_lyric_lines", "vocal_plan", "production_events",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["bpm", "meter", "duration_use", "sections"],
    "additionalProperties": False,
}

BLUEPRINT_SYSTEM_PROMPT = """Plan the musical timeline for a MiniMax Music 3 song.

Reply with exactly one JSON object and nothing else. Choose a plausible BPM and meter for the request, then create an ordered 8-18 section timeline. Every section needs a supported tag, approximate bars, target_lyric_lines, a concise vocal pacing plan, and concise production events. Instrumental, solo, wordless, and no-vocal sections use zero target lyric lines. For every other section, choose enough complete sung lines to occupy its bars at the intended vocal pacing; one lyric line normally spans roughly one or two bars, while fast rap may use more. The final section must be an outro.

The bars must fill the requested duration rather than merely describe a conventional short song. Calculate the target before assigning sections:
- 4/4 target bars = duration_seconds * bpm / 240
- 3/4 target bars = duration_seconds * bpm / 180
- 6/8 target bars = duration_seconds * bpm / 120, treating BPM as dotted-quarter beats

Use the target to choose musically sensible section proportions and line budgets; MiniJam will proportionally align the final integer bars and line targets to the exact duration after planning. Use musical judgment instead of a language-specific word or character quota. Keep the same lyrical subject and musical style as the request. Never write lyrics in this planning response."""

CAPTION_CONTRACT = """The three caption fields must follow the labeled style MiniMax Music 3 was trained on. Be concrete and musical; describe an energy arc and instrument lifecycles, never a static equipment list or decorative adjectives. Never contradict an explicit user constraint. Do not quote or paraphrase lyric lines inside the caption. Aim for roughly 250-400 words across all three fields.

`global_metadata`: one paragraph, in order: "Basic Attributes: bpm is <number>. key is <letter>, and scale is <major|minor>. <Genre / Subgenre>." then "Global Emotional Progression: <how emotion evolves from opening through the final section>." then "Application Scenarios & Imagery: <two or three vivid listening scenarios>." then "Sonics & Production Profile: <soundstage, frequency balance, dynamics, production character>."

`vocal_details`: one paragraph: "Vocal Gender & Timbre: Singer A (<Male|Female>), <timbre and register>." then "Vocal Style: <delivery and how it shifts per section>." then "Harmony/Backing Vocals: <where harmonies or doubles appear>." then "Vocal FX: <restrained reverb, delay and compression>." For an instrumental write "Instrumental, no vocals." and name the lead melodic instrument or texture.

`arrangement`: one paragraph: "Instrument Lifecycle Description (Primary/Secondary Layering): Primary: <core instruments present start to finish and their role>. Secondary: <instruments that enter, exit or intensify, and in which sections>." then "Groove & Foundation Progression: <how drums, bass and groove develop>." then "Embellishments, Textures & Spatial FX: <fills, transitions, stereo and space treatment>." Align the changes with the lyric section tags."""

LYRICS_RULES = """Use only these section tags, alone on their own line: [intro] [verse] [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] [solo] [outro]. Never put words on the same line as a tag. Roughly 12-16 sung words per 10 seconds is an upper guide. Musical directions, sound effects and stage directions never belong in the lyrics. For an instrumental, use [instrumental] with no sung words. Write lyrics in the requested language, defaulting to English."""

SYSTEM_PROMPT = f"""You write inputs for MiniMax Music 3, a lyrics-and-description music generation model.

Reply with exactly one JSON object and nothing else.

Rules:
- Separate musical direction from lyrical subject. Genre, mood, tempo, instruments, vocal style, and production terms describe how the song sounds, not what it is about.
- For a request phrased like "a <style> song about <subject>", make the title and lyrics about <subject>. Apply <style> to tags, BPM, vocals, arrangement, and production; use it in the title or lyrics only if the user explicitly makes it part of the subject.
- `title` must be a short, catchy song title.
- `tags` must be an array of 3 to 6 concise feed-card style tags.
- `bpm` must be a plausible tempo integer.
- `language` must be one of: en, zh, ja, ko, instrumental, unknown.
- `lyrics` must be a single string. {LYRICS_RULES}
- If the request is instrumental, set `language` to `instrumental`; vocal_details must never describe a singer.
- Unless explicitly instrumental, the song has a singer and vocal_details must describe that singer.
- Match the lyric length and number of sections to the requested duration and section plan.
- For non-instrumental songs, every vocal section marker must be followed by actual sung lyric lines. [instrumental] and [solo] may be empty to reserve musical space.
- Never return empty vocal sections or placeholder markers such as [END], [LYRICS], [LYRITIC], or repeated vocal labels without lyrics.
- Produce `global_metadata`, `vocal_details`, and `arrangement`. {CAPTION_CONTRACT}
- Never wrap the JSON in markdown fences.
"""


@dataclass(frozen=True)
class ComposerProfile:
    key: str
    repo_id: str
    revision: str
    filename: str
    label: str
    n_ctx: int
    max_tokens: int
    size_bytes: int


COMPOSER_PROFILES = {
    "tiny": ComposerProfile(
        key="tiny",
        repo_id="bartowski/Qwen_Qwen3.5-0.8B-GGUF",
        revision="f36b1ea49a332ede8fe5f389bbf5b3575ef71f48",
        filename="Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
        label="Qwen 3.5 0.8B Q4_K_M",
        n_ctx=4096,
        max_tokens=1800,
        size_bytes=579615840,
    ),
    "balanced": ComposerProfile(
        key="balanced",
        repo_id="bartowski/Qwen_Qwen3.5-2B-GGUF",
        revision="7d26695454df6de5fbcce2e58681e62dae06ce43",
        filename="Qwen_Qwen3.5-2B-Q4_K_M.gguf",
        label="Qwen 3.5 2B Q4_K_M",
        n_ctx=8192,
        max_tokens=1800,
        size_bytes=1396198496,
    ),
    "quality": ComposerProfile(
        key="quality",
        repo_id="bartowski/Qwen_Qwen3.5-9B-GGUF",
        revision="2dcd842c59ea5eb119267064550a7a4c592b16c3",
        filename="Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        label="Qwen 3.5 9B Q4_K_M",
        n_ctx=8192,
        max_tokens=1800,
        size_bytes=6169341984,
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
    """Extend the original Space's duration-aware lyric contract through five minutes."""
    if audio_duration > 180:
        sections = (
            "Intro", "Verse", "Pre-Chorus", "Chorus", "Instrumental", "Verse",
            "Pre-Chorus", "Chorus", "Bridge", "Solo", "Chorus", "Outro",
        )
        return {
            "structure": ", ".join(f"[{section}]" for section in sections),
            "guidance": (
                "an extended full-song arrangement with recurring verse-and-chorus cycles, "
                "bridges, a final chorus, and an outro"
            ),
            "line_range": "18 to 50",
            "min_lines": 12,
            "max_lines": 50,
            "min_words": 60,
            "max_words": round(audio_duration * 1.2),
            "target_vocal_sections": len(sections),
            "max_vocal_sections": len(sections) + 2,
            "sections": sections,
        }
    if audio_duration >= 120:
        sections = ("Verse", "Pre-Chorus", "Chorus", "Verse", "Chorus", "Bridge", "Chorus", "Outro")
        line_range, min_lines, min_words = "12 to 20", 10, 52
        guidance = "a full structure with two verses, recurring choruses, a bridge, and an outro"
    elif audio_duration >= 60:
        sections = ("Verse", "Pre-Chorus", "Chorus", "Verse", "Chorus")
        line_range, min_lines, min_words = "6 to 12", 6, 24
        guidance = "a verse, pre-chorus, chorus, second verse, and chorus reprise"
    else:
        sections = ("Verse", "Chorus")
        line_range, min_lines, min_words = "4 to 8", 4, 16
        guidance = "one concise verse and one memorable chorus"
    return {
        "structure": ", ".join(f"[{section}]" for section in sections),
        "guidance": guidance,
        "line_range": line_range,
        "min_lines": min_lines,
        "max_lines": max(min_lines + 4, round(audio_duration / 6)),
        "min_words": min_words,
        "max_words": max(36, round(audio_duration * 1.6)),
        "target_vocal_sections": len(sections),
        "max_vocal_sections": len(sections) + 2,
        "sections": sections,
    }


_BLUEPRINT_TAGS = {
    "intro", "verse", "pre-chorus", "chorus", "post-chorus",
    "bridge", "instrumental", "solo", "outro",
}


def _beats_per_bar(meter: str) -> int:
    return {"4/4": 4, "3/4": 3, "6/8": 2}.get(meter, 4)


def _target_bars(audio_duration: float, bpm: int, meter: str) -> float:
    return audio_duration * bpm / (60 * _beats_per_bar(meter))


def _requested_bpm(description: str) -> int | None:
    match = re.search(r"\b(\d{2,3})\s*bpm\b", description, re.IGNORECASE)
    if not match:
        return None
    bpm = int(match.group(1))
    return bpm if 60 <= bpm <= 180 else None


def _fit_section_bars(sections: list[dict[str, Any]], target_bars: int) -> None:
    if not 2 * len(sections) <= target_bars <= 64 * len(sections):
        raise ValueError("the selected BPM and meter cannot fit the planned section count")
    original = [section["approximate_bars"] for section in sections]
    original_total = sum(original)
    scaled = [target_bars * value / original_total for value in original]
    fitted = [max(2, min(64, int(value))) for value in scaled]
    while sum(fitted) < target_bars:
        candidates = [index for index, value in enumerate(fitted) if value < 64]
        index = max(candidates, key=lambda item: scaled[item] - fitted[item])
        fitted[index] += 1
    while sum(fitted) > target_bars:
        candidates = [index for index, value in enumerate(fitted) if value > 2]
        index = max(candidates, key=lambda item: fitted[item] - scaled[item])
        fitted[index] -= 1
    for section, old_bars, bars in zip(sections, original, fitted):
        section["approximate_bars"] = bars
        if section["target_lyric_lines"] > 0:
            scaled_lines = round(section["target_lyric_lines"] * bars / old_bars)
            section["target_lyric_lines"] = max((bars + 1) // 2, min(64, scaled_lines))


def _default_blueprint(audio_duration: float, bpm: int = 96) -> dict[str, Any]:
    meter = "4/4"
    tags = (
        "intro", "verse", "pre-chorus", "chorus", "instrumental", "verse",
        "pre-chorus", "chorus", "bridge", "solo", "chorus", "outro",
    )
    weights = (4, 12, 4, 8, 8, 12, 4, 8, 8, 8, 8, 6)
    target = max(len(tags) * 2, round(_target_bars(audio_duration, bpm, meter)))
    exact = [target * weight / sum(weights) for weight in weights]
    bars = [max(2, int(value)) for value in exact]
    while sum(bars) < target:
        index = max(range(len(bars)), key=lambda item: exact[item] - bars[item])
        bars[index] += 1
    while sum(bars) > target:
        index = max((item for item in range(len(bars)) if bars[item] > 2), key=lambda item: bars[item])
        bars[index] -= 1
    sections = []
    for tag, section_bars in zip(tags, bars):
        instrumental = tag in {"instrumental", "solo"}
        sections.append({
            "tag": tag,
            "approximate_bars": section_bars,
            "target_lyric_lines": 0 if instrumental else max(2, round(section_bars / 1.5)),
            "vocal_plan": "no lead vocal; reserve the full section" if instrumental else "concise lead phrases with breathing room",
            "production_events": "develop the arrangement through this section and transition clearly",
        })
    return {
        "bpm": bpm,
        "meter": meter,
        "duration_use": "Use the full requested timeline with an unhurried final outro.",
        "sections": sections,
    }


def _normalize_blueprint(payload: Any, audio_duration: float) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("song blueprint was not an object")
    try:
        bpm = int(payload.get("bpm"))
    except (TypeError, ValueError) as exc:
        raise ValueError("song blueprint BPM was invalid") from exc
    if not 60 <= bpm <= 180:
        raise ValueError("song blueprint BPM must be between 60 and 180")
    meter = str(payload.get("meter") or "").strip()
    if meter not in {"4/4", "3/4", "6/8"}:
        raise ValueError("song blueprint meter was invalid")
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list) or not 8 <= len(raw_sections) <= 18:
        raise ValueError("song blueprint needs 8 to 18 sections")
    sections: list[dict[str, Any]] = []
    for raw in raw_sections:
        if not isinstance(raw, dict):
            raise ValueError("song blueprint contained an invalid section")
        tag = str(raw.get("tag") or "").strip().lower().strip("[]")
        try:
            bars = int(raw.get("approximate_bars"))
            target_lines = int(raw.get("target_lyric_lines"))
        except (TypeError, ValueError) as exc:
            raise ValueError("song blueprint contained invalid bars or lyric-line targets") from exc
        if tag not in _BLUEPRINT_TAGS or not 2 <= bars <= 64 or not 0 <= target_lines <= 64:
            raise ValueError("song blueprint contained an unsupported section, bar count, or lyric-line target")
        section = {
            "tag": tag,
            "approximate_bars": bars,
            "target_lyric_lines": target_lines,
            "vocal_plan": str(raw.get("vocal_plan") or "").strip(),
            "production_events": str(raw.get("production_events") or "").strip(),
        }
        if target_lines == 0:
            if not any(
                marker in section["vocal_plan"].casefold()
                for marker in ("no vocal", "no lead", "wordless", "instrumental only", "without vocal")
            ):
                section["vocal_plan"] = f"wordless; no lead vocal. {section['vocal_plan']}".strip()
        elif not _section_requires_lyrics(section):
            section["target_lyric_lines"] = 0
        sections.append(section)
    if sections[-1]["tag"] != "outro":
        raise ValueError("song blueprint must end with an outro")
    if sum(section["target_lyric_lines"] > 0 for section in sections) < 6:
        raise ValueError("song blueprint needs at least six vocal sections")
    target = _target_bars(audio_duration, bpm, meter)
    _fit_section_bars(sections, round(target))
    return {
        "bpm": bpm,
        "meter": meter,
        "duration_use": str(payload.get("duration_use") or "").strip(),
        "sections": sections,
    }


def _blueprint_text(blueprint: dict[str, Any]) -> str:
    lines = [
        f"Tempo: {blueprint['bpm']} BPM",
        f"Meter: {blueprint['meter']}",
        f"Duration use: {blueprint['duration_use']}",
        "Exact section timeline:",
    ]
    for index, section in enumerate(blueprint["sections"], 1):
        lines.append(
            f"{index}. [{section['tag']}] - {section['approximate_bars']} bars; "
            f"target sung lines: {section['target_lyric_lines']}; "
            f"vocals: {section['vocal_plan']}; production: {section['production_events']}"
        )
    return "\n".join(lines)


def _lyrics_repair_schema(blueprint: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for index, section in enumerate(blueprint["sections"], 1):
        target_lines = section["target_lyric_lines"]
        if target_lines <= 0:
            continue
        key = f"section_{index:02d}"
        properties[key] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": target_lines,
            "maxItems": target_lines,
        }
        required.append(key)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _assemble_blueprint_lyrics(payload: Any, blueprint: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("lyric repair response was not an object")
    chunks: list[str] = []
    for index, section in enumerate(blueprint["sections"], 1):
        chunks.append(f"[{section['tag']}]")
        target_lines = section["target_lyric_lines"]
        if target_lines <= 0:
            continue
        key = f"section_{index:02d}"
        raw_lines = payload.get(key)
        if not isinstance(raw_lines, list) or len(raw_lines) != target_lines:
            raise ValueError(f"{key} did not contain exactly {target_lines} lyric lines")
        for raw_line in raw_lines:
            line = str(raw_line).replace("\r", " ").replace("\n", " ").strip()
            if not line or "[" in line or "]" in line:
                raise ValueError(f"{key} contained an empty line or section tag")
            chunks.append(line)
    return _normalize_lyrics("\n".join(chunks), instrumental=False)


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
    if section in {"Instrumental", "Solo"}:
        return []

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

    if section == "Pre-Chorus":
        variants = [
            [
                f"We feel the {accent} pulling every heartbeat into time",
                "One breath before the skyline opens wide",
            ],
            [
                f"The sound of {theme} is climbing through the room",
                "We hold the final second before everything blooms",
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


def _fallback_lyrics(
    title: str,
    description: str,
    audio_duration: float,
    instrumental: bool,
    blueprint: dict[str, Any] | None = None,
) -> str:
    if instrumental:
        return "[Instrumental]"
    hook = title or "Midnight Echo"
    terms = _subject_terms(description)
    theme = " ".join(terms[:2]).strip() or "midnight heat"
    accent = terms[2] if len(terms) >= 3 else (terms[0] if terms else "rhythm")
    plan = _lyric_plan(audio_duration)
    sections = (
        tuple(section["tag"].replace("-", " ").title().replace(" ", "-") for section in blueprint["sections"])
        if blueprint
        else plan["sections"]
    )
    section_counts: dict[str, int] = {}
    chunks: list[str] = []
    for section in sections:
        count = section_counts.get(section, 0)
        section_counts[section] = count + 1
        lines = _fallback_lines(section, count, hook, theme, accent)
        chunks.append(f"[{section}]" + ("\n" + "\n".join(lines) if lines else ""))
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
    skip_section_content = False
    for line in text.splitlines():
        match = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", line)
        if not match:
            if not skip_section_content:
                out.append(line.rstrip())
            continue
        raw_tag = re.sub(r"\s+\d+\s*$", "", match.group(1).strip().lower())
        tag = aliases.get(raw_tag, raw_tag)
        remainder = match.group(2).strip()
        if tag in allowed:
            out.append(f"[{tag}]")
            skip_section_content = tag in {"instrumental", "solo"}
            if remainder and not skip_section_content:
                out.append(remainder)
        elif remainder:
            skip_section_content = False
            out.append(remainder)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _populated_section_count(text: str) -> int:
    matches = list(re.finditer(r"(?m)^\s*\[([^\]]+)\]\s*$", text))
    populated = 0
    for index, match in enumerate(matches):
        if match.group(1).strip().lower() in {"instrumental", "solo"}:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if re.search(r"[^\W_]", text[match.end():end], re.UNICODE):
            populated += 1
    return populated


def _section_sequence(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(1).strip().lower()
        for match in re.finditer(r"(?m)^\s*\[([^\]]+)\]\s*$", text)
    )


def _section_line_counts(text: str) -> tuple[int, ...]:
    matches = list(re.finditer(r"(?m)^\s*\[([^\]]+)\]\s*$", text))
    counts = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        counts.append(len([line for line in text[match.end():end].splitlines() if line.strip()]))
    return tuple(counts)


def _section_requires_lyrics(section: dict[str, Any]) -> bool:
    if section["tag"] in {"instrumental", "solo"}:
        return False
    vocal_plan = str(section.get("vocal_plan") or "").casefold()
    silent_markers = ("no vocal", "no lead", "wordless", "instrumental only", "without vocal")
    return not any(marker in vocal_plan for marker in silent_markers)


def _lyrics_validation_error(
    text: str,
    audio_duration: float,
    blueprint: dict[str, Any] | None = None,
    language: str = "unknown",
) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ("[end]", "[lyrics]", "[lyritic]", "[end song]")):
        return "lyrics contain an unsupported placeholder tag"

    plan = _lyric_plan(audio_duration)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lyric_lines: list[str] = []
    for line in nonempty_lines:
        stripped = re.sub(r"\[[^\]]+\]", "", line).strip()
        if stripped:
            lyric_lines.append(stripped)

    if not blueprint:
        if len(lyric_lines) < plan["min_lines"]:
            return f"lyrics have {len(lyric_lines)} sung lines but need at least {plan['min_lines']}"
        if len(lyric_lines) > plan["max_lines"]:
            return f"lyrics have {len(lyric_lines)} sung lines but allow at most {plan['max_lines']}"
        body = "\n".join(lyric_lines)
        words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", body, re.UNICODE)
        if not plan["min_words"] <= len(words) <= plan["max_words"]:
            return (
                f"lyrics have {len(words)} sung words but need "
                f"{plan['min_words']} to {plan['max_words']}"
            )
    if blueprint:
        expected = tuple(section["tag"] for section in blueprint["sections"])
        actual = _section_sequence(text)
        if actual != expected:
            return f"section sequence is {actual}, but it must be exactly {expected}"
        actual_lines = _section_line_counts(text)
        for index, (section, line_count) in enumerate(zip(blueprint["sections"], actual_lines), 1):
            target_lines = section["target_lyric_lines"]
            maximum_lines = target_lines + max(2, target_lines // 2) if target_lines else 0
            if line_count < target_lines:
                return (
                    f"section {index} [{section['tag']}] has {line_count} sung lines but its blueprint "
                    f"requires at least {target_lines}"
                )
            if line_count > maximum_lines:
                return (
                    f"section {index} [{section['tag']}] has {line_count} sung lines but its blueprint "
                    f"allows at most {maximum_lines}"
                )
    elif _populated_section_count(text) > plan["max_vocal_sections"]:
        return "lyrics contain too many populated vocal sections"
    if audio_duration >= 120 and not re.search(r"(?mi)^\s*\[outro\]\s*$", text):
        return "lyrics need a final [outro] section"
    return None


def _has_meaningful_lyrics(
    text: str,
    audio_duration: float,
    blueprint: dict[str, Any] | None = None,
    language: str = "unknown",
) -> bool:
    return _lyrics_validation_error(text, audio_duration, blueprint, language) is None


def _duration_prompt(audio_duration: float, instrumental: bool) -> str:
    if instrumental:
        return "Keep the output instrumental. Use [instrumental] section tags with no sung words."
    plan = _lyric_plan(audio_duration)
    word_guidance = (
        f"Write {plan['word_range']} sung words total, following the original Space's "
        "12-16 sung words per 10 seconds rule.\n"
        if plan.get("word_range")
        else f"Stay within approximately {plan['max_words']} sung words; this is a hard maximum, not a target.\n"
    )
    return (
        f"Default structure for this duration: {plan['guidance']}.\n"
        f"Use this section plan: {plan['structure']}.\n"
        f"Write {plan['line_range']} non-empty lyric lines total.\n"
        f"{word_guidance}"
        "Keep every vocal section concise. Reserve time for instrumental sections, transitions, repeated melodic phrases, and the ending instead of filling every second with new words.\n"
        "Every vocal section must include actual sung lines. [instrumental] and [solo] may remain empty.\n"
        "Describe the same ordered section progression and its musical development in the arrangement.\n"
        "Do not emit placeholder tokens such as [END] or [LYRICS]."
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
        self.models_dir = Path(models_dir).resolve()
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
            if audio_duration > 180:
                key = "quality"
            else:
                ram_gb = _system_memory_gb()
                vram_gb = _gpu_memory_gb()
                if (vram_gb is not None and vram_gb <= 8) or (ram_gb is not None and ram_gb < 16):
                    key = "tiny"
                elif (ram_gb is not None and ram_gb >= 24) or (vram_gb is not None and vram_gb >= 16):
                    key = "quality"
                else:
                    key = "balanced"

        if key not in COMPOSER_PROFILES:
            key = "balanced"
        return COMPOSER_PROFILES[key]

    def _expand_long_lyrics(
        self,
        llm: Any,
        profile: ComposerProfile,
        lyrics: str,
        language: str,
        audio_duration: float,
        blueprint: dict[str, Any],
        progress: Callable[[str, float | None], None],
    ) -> str:
        current_lyrics = lyrics
        last_reason = _lyrics_validation_error(current_lyrics, audio_duration, blueprint, language) or "underfilled"
        repair_schema = _lyrics_repair_schema(blueprint)
        for attempt in range(1, 3):
            prompt = (
                "Repair an existing MiniMax Music 3 lyric draft. Return exactly one JSON object containing only the "
                "section arrays required by the supplied JSON schema. Each section_NN array corresponds to the same "
                "numbered vocal section in the authoritative blueprint, and the schema enforces its exact line count. "
                "Return lyric text only inside those arrays—no bracketed section tags, labels, commentary, or production "
                "directions. Keep the same language, subject, story, hook, and tone. Let the language and vocal plan "
                "determine natural phrasing and syllable density. Use the existing lyrics as creative source material, "
                "but rewrite or extend them as necessary to fulfill the complete timeline.\n\n"
                f"Language: {language}\n"
                f"Target duration: {round(audio_duration)} seconds\n"
                f"Previous validation failure: {last_reason}\n\n"
                f"Authoritative blueprint:\n{_blueprint_text(blueprint)}\n\n"
                f"Existing lyrics to preserve and expand:\n{current_lyrics}\n\n"
                "Return the complete expanded lyrics JSON now."
            )
            _log_block(f"composer.lyrics_expansion_prompt.attempt_{attempt}", prompt)
            print(f"[composer] repairing long lyrics attempt={attempt}/2...")
            progress(
                "Writing each planned lyric section…"
                if attempt == 1
                else "Correcting the planned lyric sections…"
            )
            try:
                raw = self._run_completion(
                    llm,
                    profile,
                    prompt,
                    schema=repair_schema,
                    max_tokens=3200 if profile.n_ctx >= 8192 else 2400,
                )
                _log_block(f"composer.lyrics_expansion_response.attempt_{attempt}", raw)
                expanded = _assemble_blueprint_lyrics(_extract_json(raw), blueprint)
                last_reason = _lyrics_validation_error(expanded, audio_duration, blueprint, language) or ""
                if not last_reason:
                    return expanded
            except Exception as exc:
                last_reason = str(exc) or type(exc).__name__
            print(f"[composer] lyric repair rejected attempt={attempt}/2 reason={last_reason}")
        raise ValueError(f"long-form lyric expansion failed: {last_reason}")

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
        print(f"[composer] llama-cpp-python loaded elapsed={load_elapsed:.2f}s")

        blueprint: dict[str, Any] | None = None
        if audio_duration > 180 and not instrumental:
            plan_reason = ""
            plan_prompt = (
                f"{BLUEPRINT_SYSTEM_PROMPT}\n\n"
                f"Description: {description.strip()}\n"
                f"Target duration seconds: {int(audio_duration)}\n"
                "Plan the complete timeline now."
            )
            for plan_attempt in range(1, 3):
                correction = ""
                if plan_attempt == 2:
                    correction = (
                        "\n\nCorrect the entire blueprint because validation failed: "
                        f"{plan_reason}. Recalculate the bar total from BPM, meter, and duration."
                    )
                current_prompt = f"{plan_prompt}{correction}\n\nReturn the required JSON object now."
                _log_block(f"composer.blueprint_prompt.attempt_{plan_attempt}", current_prompt)
                try:
                    print(f"[composer] generating song blueprint attempt={plan_attempt}/2...")
                    progress(
                        "Planning BPM, bars, and section timing…"
                        if plan_attempt == 1
                        else "Correcting the musical timeline…"
                    )
                    raw_plan = self._run_completion(
                        llm,
                        selected,
                        current_prompt,
                        schema=BLUEPRINT_SCHEMA,
                        max_tokens=1000,
                    )
                    _log_block(f"composer.blueprint_response.attempt_{plan_attempt}", raw_plan)
                    blueprint = _normalize_blueprint(_extract_json(raw_plan), audio_duration)
                    print(
                        "[composer] blueprint accepted "
                        f"bpm={blueprint['bpm']} meter={blueprint['meter']} "
                        f"bars={sum(section['approximate_bars'] for section in blueprint['sections'])} "
                        f"sections={len(blueprint['sections'])}"
                    )
                    break
                except Exception as exc:
                    plan_reason = str(exc) or type(exc).__name__
                    print(
                        f"[composer] blueprint rejected attempt={plan_attempt}/2 "
                        f"reason={plan_reason}"
                    )
            if blueprint is None:
                blueprint = _default_blueprint(audio_duration, bpm=_requested_bpm(description) or 96)
                print("[composer] using deterministic duration blueprint after planning retries")
            blueprint_guidance = (
                "Use the authoritative musical blueprint below. Keep its BPM, meter, exact section order, and every "
                "section occurrence. Write the target number of complete sung lyric lines declared for every section. "
                "Sections with a zero-line target must stay empty. Let the requested language and each vocal plan "
                "determine natural word and syllable density; do not substitute a generic language-wide quota. "
                "Repeated chorus lyrics must be written out in full. Describe the same timed progression in arrangement.\n\n"
                f"{_blueprint_text(blueprint)}"
            )
        else:
            blueprint_guidance = _duration_prompt(audio_duration, instrumental)

        user_prompt = (
            f"Description: {description.strip()}\n"
            f"Instrumental: {'yes' if instrumental else 'no'}\n"
            f"Target duration seconds: {int(audio_duration)}\n"
            f"{blueprint_guidance}\n"
            "Write the song spec now."
        )
        payload: dict[str, Any] = {}
        retry_reason = ""
        max_song_attempts = 2
        for attempt in range(1, max_song_attempts + 1):
            correction = ""
            if attempt > 1:
                correction = (
                    "\n\nRevise the entire JSON object because the previous attempt failed validation. "
                    f"Reason: {retry_reason}. Keep the requested concept and language, follow the exact section plan, "
                    "put every section tag on its own line, fill every vocal section with sung lines, and move all "
                    "production directions into arrangement. Return the complete corrected object."
                )
            writer_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}{correction}\n\nReturn the required JSON object now."
            _log_block(f"composer.prompt.attempt_{attempt}", writer_prompt)
            try:
                generation_started_at = time.perf_counter()
                print(f"[composer] generating song spec attempt={attempt}/{max_song_attempts}...")
                progress(
                    "Writing the title, lyrics, and production plan…"
                    if attempt == 1
                    else "Rewriting the song structure…"
                )
                content = self._run_completion(
                    llm,
                    selected,
                    writer_prompt,
                    max_tokens=2600 if blueprint else None,
                )
                generation_elapsed = time.perf_counter() - generation_started_at
                print(
                    f"[composer] completion received attempt={attempt}/{max_song_attempts} "
                    f"elapsed={generation_elapsed:.2f}s chars={len(content)}"
                )
                progress("Checking the song structure…")
                _log_block(f"composer.raw_response.attempt_{attempt}", content)
                candidate = _extract_json(content)
                candidate_lyrics = _normalize_lyrics(candidate.get("lyrics"), instrumental)
                candidate_language = str(candidate.get("language") or "").strip().lower()
                if blueprint and int(candidate.get("bpm") or 0) != blueprint["bpm"]:
                    raise ValueError(f"song BPM must remain {blueprint['bpm']} from the blueprint")
                if not instrumental and candidate_language == "instrumental":
                    raise ValueError("the writer returned instrumental mode for a vocal song")
                if not instrumental:
                    reason = _lyrics_validation_error(
                        candidate_lyrics,
                        audio_duration,
                        blueprint,
                        candidate_language,
                    )
                    if reason and blueprint:
                        candidate_lyrics = self._expand_long_lyrics(
                            llm,
                            selected,
                            candidate_lyrics,
                            candidate_language,
                            audio_duration,
                            blueprint,
                            progress,
                        )
                        candidate["lyrics"] = candidate_lyrics
                        reason = _lyrics_validation_error(
                            candidate_lyrics,
                            audio_duration,
                            blueprint,
                            candidate_language,
                        )
                    if reason:
                        raise ValueError(reason)
                payload = candidate
                print(f"[composer] parsed response keys={sorted(payload.keys())}")
                break
            except Exception as exc:
                retry_reason = str(exc) or type(exc).__name__
                print(
                    f"[composer] attempt rejected attempt={attempt}/{max_song_attempts} "
                    f"reason={retry_reason}"
                )

        if blueprint and not payload:
            self._close_llm(llm)
            raise RuntimeError(
                "The local writer could not produce lyrics dense enough for the requested long-song timeline. "
                f"Last validation error: {retry_reason}"
            )

        title = str(payload.get("title") or _guess_title(description)).strip()[:60] or "Untitled"
        tags = _normalize_tags(payload.get("tags"), description)
        bpm = blueprint["bpm"] if blueprint else payload.get("bpm")
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
        if not instrumental and (
            language == "instrumental"
            or not _has_meaningful_lyrics(lyrics, audio_duration, blueprint, language)
        ):
            language = "en"
            lyrics = _fallback_lyrics(
                title,
                description,
                audio_duration,
                instrumental=False,
                blueprint=blueprint,
            )
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
        if blueprint:
            timeline = "; ".join(
                f"[{section['tag']}] {section['approximate_bars']} bars: {section['production_events']}"
                for section in blueprint["sections"]
            )
            arrangement = (
                f"{arrangement} Planned Timeline at {blueprint['bpm']} BPM in {blueprint['meter']}: {timeline}. "
                f"Duration Use: {blueprint['duration_use']}"
            )
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

        self._close_llm(llm)

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
        expected_path = model_dir / profile.filename
        if expected_path.is_file() and expected_path.stat().st_size == profile.size_bytes:
            return expected_path
        model_path = hf_hub_download(
            repo_id=profile.repo_id,
            revision=profile.revision,
            filename=profile.filename,
            local_dir=model_dir,
        )
        return Path(model_path)

    def _load_llm(self, profile: ComposerProfile, model_path: Path):
        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed in app/env. Re-run Install or Update."
            ) from exc

        supports_gpu = bool(llama_supports_gpu_offload())
        default_gpu_layers = -1 if supports_gpu else 0
        configured_gpu_layers = os.environ.get("MINIMAX_COMPOSER_GPU_LAYERS", "").strip()
        gpu_layers = int(configured_gpu_layers) if configured_gpu_layers else default_gpu_layers
        gpu_layers = max(-1, gpu_layers)
        backend = "gpu" if supports_gpu and gpu_layers != 0 else "cpu"
        print(f"[composer] llama backend={backend} gpu_layers={gpu_layers}")
        return Llama(
            model_path=str(model_path),
            n_ctx=profile.n_ctx,
            n_batch=min(512, profile.n_ctx),
            n_gpu_layers=gpu_layers,
            n_threads=max(2, min(8, (os.cpu_count() or 4) - 1)),
            verbose=False,
        )

    @staticmethod
    def _close_llm(llm: Any) -> None:
        closer = getattr(llm, "close", None)
        if callable(closer):
            closer()
        gc.collect()

    def _run_completion(
        self,
        llm: Any,
        profile: ComposerProfile,
        prompt: str,
        *,
        schema: dict[str, Any] = SONG_SCHEMA,
        max_tokens: int | None = None,
    ) -> str:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object", "schema": schema},
            temperature=0.78,
            top_k=40,
            top_p=0.92,
            repeat_penalty=1.08,
            max_tokens=max_tokens or profile.max_tokens,
        )
        return response["choices"][0]["message"]["content"] or "{}"
