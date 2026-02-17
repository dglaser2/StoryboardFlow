"""Core creative variation pipeline."""
from __future__ import annotations

import re
import textwrap
import uuid
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

from .models import Scene, Variation
from .prompts import extract_scene, make_variation_prompts

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def run_pipeline(scene_text: str) -> Dict[str, object]:
    """Process the scene text and return payload for templates."""
    job_id = uuid.uuid4().hex[:8]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    scene = extract_scene(scene_text)
    variations = make_variation_prompts(scene)
    rendered_variations = render_images(job_dir, variations, scene)
    pdf_path = create_contact_sheet(job_dir, rendered_variations)

    return {
        "job_id": job_id,
        "scene": scene,
        "variations": rendered_variations,
        "pdf_path": f"outputs/{job_id}/{pdf_path.name}",
    }


def render_images(job_dir: Path, variations: List[Variation], scene: Scene) -> List[Variation]:
    results: List[Variation] = []
    for variation in variations:
        filename = f"{_slugify(variation.name)}.png"
        file_path = job_dir / filename
        _draw_placeholder(file_path, variation, scene)
        web_path = f"outputs/{job_dir.name}/{filename}"
        results.append(Variation(name=variation.name, prompt=variation.prompt, image_path=web_path))
    return results


def _draw_placeholder(file_path: Path, variation: Variation, scene: Scene) -> None:
    size = (1024, 576)
    img = Image.new("RGB", size, color=(14, 16, 24))
    draw = ImageDraw.Draw(img)

    header_font = _get_font(48)
    body_font = _get_font(28)
    small_font = _get_font(24)

    draw.text((40, 30), variation.name, fill=(255, 255, 255), font=header_font)

    scene_lines = [
        f"Scene: {scene.location}",
        f"Mood: {scene.mood}",
        f"Lighting: {scene.lighting}",
        f"Camera: {scene.camera}",
    ]
    draw.multiline_text((40, 120), "\n".join(scene_lines), fill=(200, 200, 210), font=body_font, spacing=6)

    prompt_block = "Prompt:\n" + _wrap_text(variation.prompt, 60)
    draw.multiline_text((40, 280), prompt_block, fill=(210, 210, 220), font=small_font, spacing=4)

    footer = "StoryboardFlow variation mode"
    draw.text((40, size[1] - 50), footer, fill=(150, 150, 160), font=small_font)

    img.save(file_path)


def create_contact_sheet(job_dir: Path, variations: List[Variation]) -> Path:
    pdf_path = job_dir / "contact_sheet.pdf"
    page_width, page_height = landscape(letter)
    c = CanvasWrapper(str(pdf_path), page_width, page_height)

    cols = 2
    rows = 2
    margin = 0.5 * inch
    tile_width = (page_width - (margin * 2)) / cols - 0.25 * inch
    tile_height = (page_height - (margin * 2)) / rows - 0.5 * inch

    for idx, variation in enumerate(variations):
        row = idx // cols
        col = idx % cols
        x = margin + col * (tile_width + 0.25 * inch)
        y = page_height - margin - (row + 1) * (tile_height + 0.5 * inch)
        img_path = BASE_DIR / variation.image_path
        reader = ImageReader(str(img_path))
        c.drawImage(reader, x, y + 0.25 * inch, width=tile_width, height=tile_height, preserveAspectRatio=True, anchor="c")
        caption = f"{variation.name}: {variation.prompt[:80]}"
        c.drawString(x, y, caption)

    c.save()
    return pdf_path


class CanvasWrapper:
    """Light wrapper around reportlab canvas for easier testing."""

    def __init__(self, path: str, width: float, height: float) -> None:
        from reportlab.pdfgen import canvas

        self._canvas = canvas.Canvas(path, pagesize=(width, height))

    def drawImage(self, *args, **kwargs):
        self._canvas.drawImage(*args, **kwargs)

    def drawString(self, *args, **kwargs):
        self._canvas.setFont("Helvetica", 12)
        self._canvas.drawString(*args, **kwargs)

    def save(self) -> None:
        self._canvas.showPage()
        self._canvas.save()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "variation"


def _wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _get_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

# Ensure base output directory exists on import
OUTPUT_DIR.mkdir(exist_ok=True)
