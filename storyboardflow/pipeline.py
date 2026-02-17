"""Pipeline logic for storyboard branching workflow."""
from __future__ import annotations

import base64
import mimetypes
import os
import shutil
import subprocess
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from dataclasses import dataclass

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from PIL import Image, ImageStat

from .models import Frame, JobState, load_state, save_state

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOADS_NAME = "uploads"
CLIPS_NAME = "clips"
LOOKAHEAD_WINDOW = 2
FFMPEG_BIN = shutil.which("ffmpeg")

LIGHTING_OPTIONS = ["warm sunset", "neon noir", "high contrast", "soft studio"]
CAMERA_OPTIONS = ["close-up", "over-the-shoulder", "low angle", "tracking"]


@dataclass
class UploadResult:
    job_id: str
    redirect_url: str


class Captioner:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client: Optional[object] = None
        self.legacy_module = None
        self.use_responses = False
        if self.api_key:
            try:  # new SDK
                from openai import OpenAI

                self.client = OpenAI()
                self.use_responses = bool(getattr(self.client, "responses", None))
            except Exception:
                self.client = None
            if not self.use_responses:
                try:
                    import openai

                    openai.api_key = self.api_key
                    self.legacy_module = openai
                except Exception:
                    self.legacy_module = None

    def caption(self, image_path: Path, idx: int) -> str:
        if self.client and self.use_responses:
            try:
                return self._caption_with_openai(image_path)
            except Exception as exc:
                print(f"[captioner] OpenAI responses fallback: {exc}")
        if self.client and not self.use_responses:
            try:
                return self._caption_with_images(image_path)
            except Exception as exc:
                print(f"[captioner] images endpoint fallback: {exc}")
        return self._fallback_caption(image_path, idx)

    def _caption_with_openai(self, image_path: Path) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client unavailable")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        with image_path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this storyboard frame in one vivid sentence."},
                        {
                            "type": "input_image",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.2,
        )
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "text":
                    return content.text.strip()
        raise ValueError("No caption returned")

    def _caption_with_images(self, image_path: Path) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client unavailable")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        with image_path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this storyboard frame in one vivid sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ],
                }
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _fallback_caption(self, image_path: Path, idx: int) -> str:
        try:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                stat = ImageStat.Stat(rgb)
                r, g, b = stat.mean
                brightness = (r + g + b) / 3
                palette = "cool" if b > r + 8 else "warm" if r > b + 8 else "neutral"
                light = "low-light" if brightness < 60 else "soft" if brightness < 150 else "bright"
                aspect = rgb.width / rgb.height if rgb.height else 1
                framing = "wide" if aspect >= 1.2 else "tall" if aspect <= 0.8 else "balanced"
                return f"Frame {idx + 1} {framing} shot, {palette} palette, {light} fallback capture"
        except Exception:
            pass
        return f"Storyboard frame {idx + 1}: (no vision key)"


captioner = Captioner()


def get_job_dir(job_id: str) -> Path:
    return OUTPUT_DIR / job_id


def get_state_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "state.json"


def ensure_state(job_id: str) -> JobState:
    path = get_state_path(job_id)
    state = load_state(path)
    if not state:
        raise FileNotFoundError(f"Job {job_id} not found")
    state.compute_staleness()
    return state


def create_job(files: Sequence[Tuple[str, bytes]]) -> UploadResult:
    if not 4 <= len(files) <= 8:
        raise ValueError("Upload between 4 and 8 frames")

    job_id = uuid.uuid4().hex[:8]
    job_dir = get_job_dir(job_id)
    uploads_dir = job_dir / UPLOADS_NAME
    clips_dir = job_dir / CLIPS_NAME
    uploads_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(exist_ok=True)

    frames: List[Frame] = []
    constraints = ""

    for idx, (filename, data) in enumerate(files):
        suffix = Path(filename).suffix or ".png"
        upload_path = uploads_dir / f"frame_{idx+1:02d}{suffix}"
        with upload_path.open("wb") as f:
            f.write(data)
        caption = captioner.caption(upload_path, idx)
        prompts = build_prompts(caption, constraints)
        frame = Frame(
            index=idx,
            upload_path=str(relative_to_outputs(upload_path)),
            caption=caption,
            prompt_base=prompts[0],
            prompt_altA=prompts[1],
            prompt_altB=prompts[2],
            base_clip_path=str(render_variant(upload_path, clips_dir, idx, "base", prompts[0])),
            altA_clip_path=str(render_variant(upload_path, clips_dir, idx, "altA", prompts[1])),
            altB_clip_path=str(render_variant(upload_path, clips_dir, idx, "altB", prompts[2])),
            constraints_version=0,
        )
        frames.append(frame)

    state = JobState(
        job_id=job_id,
        num_frames=len(frames),
        global_constraints=constraints,
        frames=frames,
    )
    save_state(state, get_state_path(job_id))
    return UploadResult(job_id=job_id, redirect_url=f"/review/{job_id}")


def build_prompts(caption: str, constraints: str) -> Tuple[str, str, str]:
    base = _clamp(f"{caption}. {constraints} cinematic storyboard still")
    idx = sum(ord(c) for c in caption)
    lighting = LIGHTING_OPTIONS[idx % len(LIGHTING_OPTIONS)]
    camera = CAMERA_OPTIONS[idx % len(CAMERA_OPTIONS)]
    altA = _clamp(f"{caption}. {constraints} with {lighting} lighting cinematic still")
    altB = _clamp(f"{caption}. {constraints} shot as {camera} cinematic still")
    return base, altA, altB


def render_variant(upload_path: Path, clips_dir: Path, idx: int, variant: str, prompt: str) -> Path:
    file_name = f"frame_{idx+1:02d}_{variant}.mp4"
    out_path = clips_dir / file_name
    overlay = f"{variant.upper()}: {prompt[:80]}"
    if render_preview_clip(upload_path, out_path, overlay):
        return relative_to_outputs(out_path)
    return relative_to_outputs(upload_path)


def render_preview_clip(image_path: Path, out_mp4: Path, overlay_text: str) -> bool:
    if not FFMPEG_BIN:
        return False
    safe_text = overlay_text.replace(":", "-").replace("'", "\'").replace("\n", " ")[:120]
    filter_chain = (
        "zoompan=z='min(zoom+0.002,1.15)':d=75:s=1024x576,"  # Ken Burns
        "drawtext=text='" + safe_text + "':fontcolor=white:fontsize=32:x=40:y=40:box=1:boxcolor=0x000000AA"
    )
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-vf",
        filter_chain,
        "-c:v",
        "libx264",
        "-t",
        "2.5",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def relative_to_outputs(path: Path) -> Path:
    return Path("outputs") / path.relative_to(OUTPUT_DIR)


def choose_variant(job_id: str, frame_index: int, choice: str) -> JobState:
    if choice not in {"base", "altA", "altB"}:
        raise ValueError("Invalid choice")
    state = ensure_state(job_id)
    if frame_index < 0 or frame_index >= state.num_frames:
        raise ValueError("Frame index out of range")

    frame = state.frames[frame_index]
    frame.chosen = choice
    state.constraints_version += 1
    state.global_constraints = describe_constraints(frame_index, choice)

    job_dir = get_job_dir(job_id)
    clips_dir = job_dir / CLIPS_NAME
    clips_dir.mkdir(exist_ok=True)

    for idx in range(frame_index + 1, state.num_frames):
        fr = state.frames[idx]
        prompts = build_prompts(fr.caption, state.global_constraints)
        fr.prompt_base, fr.prompt_altA, fr.prompt_altB = prompts
        if idx <= frame_index + LOOKAHEAD_WINDOW:
            upload = BASE_DIR / Path(fr.upload_path)
            fr.base_clip_path = str(render_variant(upload, clips_dir, idx, "base", fr.prompt_base))
            fr.altA_clip_path = str(render_variant(upload, clips_dir, idx, "altA", fr.prompt_altA))
            fr.altB_clip_path = str(render_variant(upload, clips_dir, idx, "altB", fr.prompt_altB))
            fr.constraints_version = state.constraints_version
        else:
            fr.constraints_version = state.constraints_version - 1

    state.frames[frame_index].constraints_version = state.constraints_version
    state.branch_events.append(
        {
            "frame_index": frame_index,
            "chosen": choice,
            "constraints_version": state.constraints_version,
            "global_constraints": state.global_constraints,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    save_state(state, get_state_path(job_id))
    return state


def describe_constraints(frame_index: int, choice: str) -> str:
    frame_num = frame_index + 1
    if choice == "altA":
        return f"Constraints: lock lighting mood chosen at frame {frame_num}"
    if choice == "altB":
        return f"Constraints: adopt tighter camera coverage from frame {frame_num}"
    return f"Constraints: preserve baseline look after frame {frame_num}"


def regen_frame(job_id: str, frame_index: int) -> JobState:
    state = ensure_state(job_id)
    if frame_index < 0 or frame_index >= state.num_frames:
        raise ValueError("Frame index out of range")
    frame = state.frames[frame_index]
    if frame.constraints_version >= state.constraints_version:
        return state

    prompts = build_prompts(frame.caption, state.global_constraints)
    frame.prompt_base, frame.prompt_altA, frame.prompt_altB = prompts
    job_dir = get_job_dir(job_id)
    clips_dir = job_dir / CLIPS_NAME
    clips_dir.mkdir(exist_ok=True)
    upload = BASE_DIR / Path(frame.upload_path)
    frame.base_clip_path = str(render_variant(upload, clips_dir, frame_index, "base", frame.prompt_base))
    frame.altA_clip_path = str(render_variant(upload, clips_dir, frame_index, "altA", frame.prompt_altA))
    frame.altB_clip_path = str(render_variant(upload, clips_dir, frame_index, "altB", frame.prompt_altB))
    frame.constraints_version = state.constraints_version

    save_state(state, get_state_path(job_id))
    return state


def export_pdf(job_id: str) -> Path:
    state = ensure_state(job_id)
    job_dir = get_job_dir(job_id)
    pdf_path = job_dir / "contact_sheet.pdf"
    page_width, page_height = landscape(letter)
    c = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height))

    cols = 3
    rows = max(1, -(-state.num_frames // cols))
    tile_width = (page_width - inch) / cols
    tile_height = (page_height - inch) / rows

    for idx, frame in enumerate(state.frames):
        col = idx % cols
        row = idx // cols
        x = 0.5 * inch + col * tile_width
        y = page_height - 0.5 * inch - (row + 1) * tile_height
        source = BASE_DIR / Path(frame.upload_path)
        if source.exists():
            c.drawImage(ImageReader(str(source)), x, y + 0.4 * inch, tile_width - 0.2 * inch, tile_height - inch, preserveAspectRatio=True)
        caption = f"Frame {frame.index+1} choice: {frame.chosen}"
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y + 0.2 * inch, caption)
        c.setFont("Helvetica", 10)
        c.drawString(x, y, frame.caption[:100])
    c.showPage()
    c.save()
    return Path(f"outputs/{job_id}/contact_sheet.pdf")


def _clamp(text: str, limit: int = 220) -> str:
    return textwrap.shorten(text, width=limit, placeholder="...")


OUTPUT_DIR.mkdir(exist_ok=True)
