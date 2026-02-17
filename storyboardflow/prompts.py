"""Scene extraction and prompt generation utilities."""
from __future__ import annotations

import json
import os
import re
import textwrap
from typing import List

from .models import Scene, Variation

try:  # Optional OpenAI import
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


LOCATION_HINTS = {
    "subway": "crowded subway car",
    "metro": "underground subway platform",
    "train": "commuter train interior",
    "cafe": "intimate cafe corner",
    "coffee": "modern cafe bar",
    "office": "glass-walled tech office",
    "street": "rain-soaked city street",
    "apartment": "loft apartment living room",
    "forest": "misty pine forest clearing",
    "alley": "narrow neon alley",
    "studio": "artist loft studio",
}

MOOD_HINTS = {
    "tense": "tense",
    "nervous": "anxious",
    "calm": "calm",
    "joy": "joyful",
    "happy": "hopeful",
    "hope": "hopeful",
    "melancholy": "melancholy",
    "sad": "somber",
    "angry": "intense",
    "determined": "determined",
    "uneasy": "uneasy",
}

LIGHTING_VARIANTS = [
    "warm sunset glow",
    "moody neon rim light",
    "high-contrast noir shadows",
    "misty morning diffusion",
]

CAMERA_VARIANTS = [
    "intimate over-the-shoulder focus",
    "dynamic low angle close-up",
    "sweeping crane shot",
    "handheld documentary framing",
]

MOOD_VARIANTS = [
    "resolute",
    "uneasy",
    "hopeful",
    "brooding",
]


def extract_scene(scene_text: str) -> Scene:
    """Return a structured Scene from text via LLM or heuristics."""
    cleaned = scene_text.strip()
    if not cleaned:
        cleaned = "A lone protagonist waits for the train in a dim station."  # default prompt

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and OpenAI is not None:
        try:
            return _extract_scene_with_openai(cleaned)
        except Exception:
            # Fall through to heuristics on any error
            pass

    return _extract_scene_heuristic(cleaned)


def _extract_scene_with_openai(scene_text: str) -> Scene:
    client = OpenAI()
    schema = {
        "title": "Short descriptive title",
        "characters": ["list of key characters"],
        "location": "setting",
        "time_of_day": "day, night, etc.",
        "mood": "emotional tone",
        "lighting": "lighting description",
        "camera": "camera framing",
        "style": "visual treatment",
        "constraints": ["continuity reminders"],
    }
    prompt = (
        "You are a storyboard director. Read the scene description and respond "
        "with strict JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "The JSON must be parseable and include concise but vivid phrases."
    )
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": scene_text},
        ],
        temperature=0.4,
    )

    text_parts: List[str] = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "text":
                text_parts.append(content.text)
    raw = "\n".join(text_parts).strip()
    try:
        data = json.loads(raw)
        return Scene(**data)
    except Exception as exc:
        raise ValueError("Failed to parse LLM scene JSON") from exc


def _extract_scene_heuristic(scene_text: str) -> Scene:
    lower = scene_text.lower()
    sentences = re.split(r"(?<=[.!?])\s+", scene_text)
    title = sentences[0].strip().rstrip(".!") or "Untitled moment"

    name_candidates = re.findall(r"\b([A-Z][a-z]{2,})\b", scene_text)
    ignore = {"A", "The", "At", "In", "On", "And"}
    characters = [name for name in name_candidates if name not in ignore]
    if not characters:
        characters = ["Protagonist"]

    location = "Unknown location"
    for keyword, loc in LOCATION_HINTS.items():
        if keyword in lower:
            location = loc
            break

    if "night" in lower or "midnight" in lower or "dusk" in lower or "evening" in lower:
        time_of_day = "night"
    elif "sunrise" in lower or "morning" in lower or "dawn" in lower:
        time_of_day = "morning"
    elif "afternoon" in lower or "day" in lower:
        time_of_day = "day"
    else:
        time_of_day = "unspecified"

    mood = "neutral"
    for keyword, value in MOOD_HINTS.items():
        if keyword in lower:
            mood = value
            break

    lighting = "natural daylight"
    if time_of_day == "night" or "fluorescent" in lower or "office" in lower:
        lighting = "cold fluorescent wash"
    if "sunset" in lower:
        lighting = "warm sunset glow"

    camera = "wide establishing"
    if "close" in lower:
        camera = "intimate close-up"

    style = "cinematic storyboard still"

    constraints = [
        f"Maintain {location} setting across panels",
        f"Keep characters {' and '.join(characters[:2])} consistent",
        f"Preserve {mood} mood unless otherwise noted",
    ]

    return Scene(
        title=title,
        characters=characters,
        location=location,
        time_of_day=time_of_day,
        mood=mood,
        lighting=lighting,
        camera=camera,
        style=style,
        constraints=constraints[:3],
    )


def make_variation_prompts(scene: Scene) -> List[Variation]:
    """Create four deterministic prompts based on the scene."""
    base = _describe_scene(scene)
    lighting = _pick_variant(LIGHTING_VARIANTS, scene.lighting)
    camera = _pick_variant(CAMERA_VARIANTS, scene.camera)
    mood = _pick_variant(MOOD_VARIANTS, scene.mood)

    variations = [
        Variation(name="Faithful Translation", prompt=_clamp(base), image_path=""),
        Variation(
            name="Lighting Variation",
            prompt=_clamp(
                _describe_scene(scene, lighting=lighting)
                + " Emphasize the transformed lighting while preserving blocking."
            ),
            image_path="",
        ),
        Variation(
            name="Camera Variation",
            prompt=_clamp(
                _describe_scene(scene, camera=camera)
                + " Adjust staging for the new lens movement only."
            ),
            image_path="",
        ),
        Variation(
            name="Emotion Variation",
            prompt=_clamp(
                _describe_scene(scene, mood=mood)
                + " Shift expressions and posture to reflect the altered emotion."
            ),
            image_path="",
        ),
    ]
    return variations


def _describe_scene(scene: Scene, lighting: str | None = None, camera: str | None = None, mood: str | None = None) -> str:
    characters = ", ".join(scene.characters)
    constraints = "; ".join(scene.constraints)
    description = (
        f"{scene.title} rendered as a {scene.style}. Location: {scene.location}. "
        f"Time: {scene.time_of_day}. Characters: {characters}. Mood: {mood or scene.mood}. "
        f"Lighting: {lighting or scene.lighting}. Camera: {camera or scene.camera}. "
        f"Continuity notes: {constraints}."
    )
    return description


def _pick_variant(options: List[str], current: str) -> str:
    if not options:
        return current
    index = sum(ord(ch) for ch in current) % len(options)
    variant = options[index]
    if variant == current:
        variant = options[(index + 1) % len(options)]
    return variant


def _clamp(text: str, limit: int = 480) -> str:
    clipped = textwrap.shorten(text, width=limit, placeholder="...")
    return clipped
