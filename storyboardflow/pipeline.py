"""Pipeline logic for storyboard branching workflow."""
from __future__ import annotations

import base64
import mimetypes
import os
import shutil
import subprocess
import textwrap
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .models import Frame, JobState, load_state, save_state

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOADS_NAME = "uploads"
ENHANCED_NAME = "enhanced"
STILLS_NAME = "stills"
CLIPS_NAME = "clips"
VIDEOS_NAME = "videos"
LOOKAHEAD_WINDOW = 2
FFMPEG_BIN = shutil.which("ffmpeg")

LIGHTING_OPTIONS = ["warm sunset", "neon noir", "high contrast", "soft studio"]
CAMERA_OPTIONS = ["close-up", "over-the-shoulder", "low angle", "tracking"]
LIGHTING_CAPTION_SUFFIX = "dramatic lighting emphasis"
CAMERA_CAPTION_SUFFIX = "tighter narrative camera"
VIDEO_PROMPT_SUFFIX = "cinematic 3 second shot, smooth motion, professional film"
VEO_MODEL = "veo-3.1-generate-preview"
VEO_ASPECT_RATIO = "16:9"
VEO_RESOLUTION = "720p"
VEO_DURATION_SECONDS = 3
VEO_POLL_INTERVAL = 10
VEO_MAX_WAIT_SECONDS = 600

try:  # Optional import for Gemini client
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    genai_types = None  # type: ignore

VARIANT_META = {
    "base": {
        "still_attr": "still_base_path",
        "prompt_attr": "prompt_base",
        "clip_attr": "base_clip_path",
        "video_attr": "video_base_path",
        "status_attr": "video_base_status",
        "job_attr": "video_base_job_id",
        "provider_attr": "video_base_provider",
    },
    "altA": {
        "still_attr": "still_altA_path",
        "prompt_attr": "prompt_altA",
        "clip_attr": "altA_clip_path",
        "video_attr": "video_altA_path",
        "status_attr": "video_altA_status",
        "job_attr": "video_altA_job_id",
        "provider_attr": "video_altA_provider",
    },
    "altB": {
        "still_attr": "still_altB_path",
        "prompt_attr": "prompt_altB",
        "clip_attr": "altB_clip_path",
        "video_attr": "video_altB_path",
        "status_attr": "video_altB_status",
        "job_attr": "video_altB_job_id",
        "provider_attr": "video_altB_provider",
    },
}


@dataclass
class UploadResult:
    job_id: str
    redirect_url: str


class Captioner:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client: Optional[object] = None
        self.use_responses = False
        if self.api_key:
            try:  # new SDK
                from openai import OpenAI

                self.client = OpenAI()
                self.use_responses = bool(getattr(self.client, "responses", None))
            except Exception:
                self.client = None

    def caption(self, image_path: Path, frame_number: int) -> str:
        if self.client and self.use_responses:
            try:
                return self._caption_with_responses(image_path)
            except Exception as exc:
                print(f"[captioner] responses fallback: {exc}")
                try:
                    return self._caption_with_chat(image_path)
                except Exception as exc_chat:
                    print(f"[captioner] chat fallback after responses error: {exc_chat}")
        elif self.client:
            try:
                return self._caption_with_chat(image_path)
            except Exception as exc:
                print(f"[captioner] chat fallback: {exc}")
        return f"Storyboard frame {frame_number}: (no vision key)."

    def _caption_with_responses(self, image_path: Path) -> str:
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
                        {
                            "type": "input_text",
                            "text": "Describe this storyboard frame in one vivid sentence.",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{b64}",
                        },
                    ],
                }
            ],
            max_output_tokens=64,
            temperature=0.1,
        )
        for candidate in getattr(response, "output", []):
            for content in getattr(candidate, "content", []):
                if getattr(content, "type", None) == "output_text" and hasattr(content, "text"):
                    text = content.text.strip()
                    if text:
                        return text
                if getattr(content, "type", None) == "text":  # older SDKs
                    text = content.text.strip()
                    if text:
                        return text
        raise ValueError("No caption returned")

    def _caption_with_chat(self, image_path: Path) -> str:
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


captioner = Captioner()


def caption_frame(image_path: Path, frame_number: int) -> str:
    try:
        start = time.monotonic()
        caption = captioner.caption(image_path, frame_number)
        elapsed = time.monotonic() - start
        print(f"[captioner] frame {frame_number} completed in {elapsed:.2f}s")
        return caption
    except Exception as exc:
        print(f"[captioner] fallback heuristic for frame {frame_number}: {exc}")
        return f"Storyboard frame {frame_number}: (no vision key)."


def _veo_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _veo_enabled() -> bool:
    return bool(_veo_api_key() and genai is not None)


def _veo_client():
    api_key = _veo_api_key()
    if not api_key or genai is None:
        raise RuntimeError("Gemini API key or google-genai client unavailable")
    return genai.Client(api_key=api_key)


def _variant_fields(variant: str) -> Dict[str, str]:
    if variant not in VARIANT_META:
        raise ValueError(f"Unknown variant {variant}")
    return VARIANT_META[variant]


def _invalidate_videos(frame: Frame, variants: Sequence[str]) -> None:
    for variant in variants:
        fields = _variant_fields(variant)
        setattr(frame, fields["video_attr"], None)
        setattr(frame, fields["status_attr"], "missing")
        setattr(frame, fields["job_attr"], None)
        setattr(frame, fields.get("provider_attr"), None)


def enhance_frame(in_path: Path, out_path: Path, target_width: int = 1280) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(in_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = target_width / width
        new_height = max(1, int(height * scale))
        resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        contrast = ImageEnhance.Contrast(resized).enhance(1.12)
        sharp = ImageEnhance.Sharpness(contrast).enhance(1.08)
        sharp.save(out_path, format="PNG")
    return out_path


def make_variant_stills(
    enhanced_path: Path,
    out_dir: Path,
    frame_number: int,
    base_caption: str,
    lighting_caption: str,
    camera_caption: str,
    global_constraints: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(enhanced_path) as base_img:
        base_img = base_img.convert("RGB")
        base_caption = _sanitize_caption(base_caption)
        lighting_caption = _sanitize_caption(lighting_caption)
        camera_caption = _sanitize_caption(camera_caption)

        base_still = _annotate_variant(base_img.copy(), base_caption, f"Beat {frame_number} · Base", accent=(125, 185, 255))
        base_path = out_dir / f"frame_{frame_number:02d}_base.png"
        base_still.save(base_path)

        lighting_image = _apply_lighting_grade(base_img.copy(), frame_number, global_constraints)
        lighting_still = _annotate_variant(lighting_image, lighting_caption, "Alt A · Lighting", accent=(255, 170, 110))
        lighting_path = out_dir / f"frame_{frame_number:02d}_altA.png"
        lighting_still.save(lighting_path)

        camera_image = _apply_camera_variant(base_img.copy(), frame_number)
        camera_still = _annotate_variant(camera_image, camera_caption, "Alt B · Camera", accent=(170, 200, 255))
        camera_path = out_dir / f"frame_{frame_number:02d}_altB.png"
        camera_still.save(camera_path)

    return {"base": base_path, "altA": lighting_path, "altB": camera_path}

def _annotate_variant(image: Image.Image, caption: str, label: str, accent: Tuple[int, int, int]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    overlay_height = int(height * 0.1)
    base_rect = Image.new("RGBA", (width, overlay_height), (4, 6, 12, 220))
    accent_bar = Image.new("RGBA", (width, 6), accent + (255,))
    image.paste(accent_bar, (0, height - overlay_height - 6), accent_bar)
    image.paste(base_rect, (0, height - overlay_height), base_rect)
    draw.text((24, height - overlay_height + 8), _sanitize_caption(label)[:60], fill=accent)
    draw.text((24, height - overlay_height + 40), _sanitize_caption(caption)[:68], fill=(230, 230, 230))
    return image


def _variant_caption(base_caption: str, suffix: str) -> str:
    short = base_caption.strip().rstrip(".")
    return _sanitize_caption(f"{short} — {suffix}")


def _video_prompt(prompt: str) -> str:
    combined = f"{prompt} {VIDEO_PROMPT_SUFFIX}".strip()
    if len(combined) > 300:
        combined = combined[:297] + "..."
    return combined


def _apply_lighting_grade(image: Image.Image, frame_number: int, constraints: str) -> Image.Image:
    original = image.copy()
    constraints_lower = constraints.lower()
    warm_bias = "warm" in constraints_lower or "sunset" in constraints_lower or (frame_number % 2 == 0)
    if warm_bias:
        image = ImageEnhance.Color(image).enhance(1.35)
        image = ImageEnhance.Brightness(image).enhance(1.1)
        overlay = Image.new("RGB", image.size, (255, 180, 90))
        image = Image.blend(image, overlay, 0.18)
    else:
        image = ImageEnhance.Color(image).enhance(0.75)
        image = ImageEnhance.Contrast(image).enhance(1.25)
        image = ImageEnhance.Brightness(image).enhance(0.92)
        overlay = Image.new("RGB", image.size, (80, 150, 255))
        image = Image.blend(image, overlay, 0.2)
    vignette = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(vignette)
    w, h = image.size
    draw.ellipse((-w * 0.2, -h * 0.2, w * 1.2, h * 1.2), fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=min(w, h) * 0.2))
    vignette_mask = ImageOps.invert(vignette).point(lambda p: p * 0.6)
    dark_layer = Image.new("RGB", image.size, color=(5, 5, 5))
    graded = Image.composite(image, dark_layer, vignette_mask)
    return Image.blend(graded, original, 0.35)


def _apply_camera_variant(image: Image.Image, frame_number: int) -> Image.Image:
    width, height = image.size
    zoom = 0.65
    crop_w = int(width * zoom)
    crop_h = int(height * zoom)
    x_offset = (width - crop_w) // 2
    if frame_number % 2 == 0:
        x_offset = min(width - crop_w, x_offset + int(width * 0.12))
    else:
        x_offset = max(0, x_offset - int(width * 0.12))
    y_offset = (height - crop_h) // 2
    crop = image.crop((x_offset, y_offset, x_offset + crop_w, y_offset + crop_h))
    zoomed = crop.resize((width, height), Image.Resampling.LANCZOS)
    mask = Image.new("L", zoomed.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([(0, int(height * 0.2)), (width, int(height * 0.8))], fill=180)
    bars = Image.new("RGB", zoomed.size, color=(5, 5, 8))
    cinematic = Image.composite(zoomed, bars, mask)
    return ImageEnhance.Sharpness(cinematic).enhance(1.25)


def render_preview_clip(still_path: Path, out_mp4: Path, overlay_text: str) -> bool:
    if not FFMPEG_BIN:
        return False
    safe_text = _safe_overlay_text(overlay_text)
    filter_chain = (
        "zoompan=z='min(zoom+0.002,1.12)':d=75:s=1024x576,"
        "drawtext=text='" + safe_text + "':fontcolor=white:fontsize=30:x=40:y=40:box=1:boxcolor=0x00000088"
    )
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-loop",
        "1",
        "-i",
        str(still_path),
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


def get_job_dir(job_id: str) -> Path:
    return OUTPUT_DIR / job_id


def get_state_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "state.json"


def _videos_dir(job_id: str) -> Path:
    path = get_job_dir(job_id) / VIDEOS_NAME
    path.mkdir(exist_ok=True)
    return path


def ensure_state(job_id: str) -> JobState:
    path = get_state_path(job_id)
    state = load_state(path)
    if not state:
        print(f"[state] missing job {job_id} at {path}")
        raise FileNotFoundError(f"Job {job_id} not found")
    state.compute_staleness()
    return state


def create_job(files: Sequence[Tuple[str, bytes]]) -> UploadResult:
    if not 4 <= len(files) <= 8:
        raise ValueError("Upload between 4 and 8 frames")

    job_id = uuid.uuid4().hex[:8]
    job_dir = get_job_dir(job_id)
    uploads_dir = job_dir / UPLOADS_NAME
    enhanced_dir = job_dir / ENHANCED_NAME
    stills_dir = job_dir / STILLS_NAME
    clips_dir = job_dir / CLIPS_NAME
    videos_dir = job_dir / VIDEOS_NAME
    uploads_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(exist_ok=True)
    stills_dir.mkdir(exist_ok=True)
    clips_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    frames: List[Frame] = []
    constraints = ""

    uploads: List[Tuple[int, Path]] = []
    for idx, (filename, data) in enumerate(files):
        frame_number = idx + 1
        suffix = Path(filename).suffix or ".png"
        upload_path = uploads_dir / f"frame_{frame_number:02d}{suffix}"
        with upload_path.open("wb") as f:
            f.write(data)
        uploads.append((frame_number, upload_path))

    captions = _collect_captions_parallel(uploads)

    for idx, (frame_number, upload_path) in enumerate(uploads):
        caption = captions.get(frame_number, f"Storyboard frame {frame_number}: (no vision key).")
        lighting_caption = _variant_caption(caption, LIGHTING_CAPTION_SUFFIX)
        camera_caption = _variant_caption(caption, CAMERA_CAPTION_SUFFIX)
        enhanced_path = enhance_frame(upload_path, enhanced_dir / f"frame_{frame_number:02d}_enhanced.png")
        still_paths = make_variant_stills(
            enhanced_path,
            stills_dir,
            frame_number,
            caption,
            lighting_caption,
            camera_caption,
            constraints,
        )
        prompts = build_prompts(caption, constraints)
        clip_paths = _render_clips_from_stills(frame_number, still_paths, clips_dir)

        frame = Frame(
            index=idx,
            upload_path=str(relative_to_outputs(upload_path)),
            enhanced_path=str(relative_to_outputs(enhanced_path)),
            still_base_path=str(relative_to_outputs(still_paths["base"])),
            still_altA_path=str(relative_to_outputs(still_paths["altA"])),
            still_altB_path=str(relative_to_outputs(still_paths["altB"])),
            caption=caption,
            caption_altA=lighting_caption,
            caption_altB=camera_caption,
            prompt_base=prompts[0],
            prompt_altA=prompts[1],
            prompt_altB=prompts[2],
            base_clip_path=clip_paths["base"],
            altA_clip_path=clip_paths["altA"],
            altB_clip_path=clip_paths["altB"],
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


def _collect_captions_parallel(uploads: List[Tuple[int, Path]]) -> Dict[int, str]:
    captions: Dict[int, str] = {}
    if not uploads:
        return captions
    max_workers = min(4, len(uploads))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_safe_caption_frame, frame_number, path): frame_number
            for frame_number, path in uploads
        }
        for future in as_completed(futures):
            frame_number = futures[future]
            try:
                captions[frame_number] = future.result()
            except Exception as exc:
                print(f"[captioner] executor fallback for frame {frame_number}: {exc}")
                captions[frame_number] = f"Storyboard frame {frame_number}: (no vision key)."
    return captions


def _safe_caption_frame(frame_number: int, path: Path) -> str:
    caption = caption_frame(path, frame_number)
    return _sanitize_caption(caption)


def build_prompts(caption: str, constraints: str) -> Tuple[str, str, str]:
    base = _clamp(f"{caption}. {constraints} cinematic storyboard still")
    idx = sum(ord(c) for c in caption)
    lighting = LIGHTING_OPTIONS[idx % len(LIGHTING_OPTIONS)]
    camera = CAMERA_OPTIONS[idx % len(CAMERA_OPTIONS)]
    altA = _clamp(f"{caption}. {constraints} with {lighting} lighting cinematic still")
    altB = _clamp(f"{caption}. {constraints} shot as {camera} cinematic still")
    return base, altA, altB


def generate_video_async(job_id: str, frame_index: int, variant: str) -> None:
    if variant not in VARIANT_META:
        raise ValueError("Unknown variant")
    if not _veo_enabled():
        raise RuntimeError("Gemini Veo API key not configured")
    state = ensure_state(job_id)
    frame = state.frames[frame_index]
    fields = _variant_fields(variant)
    status = getattr(frame, fields["status_attr"], "missing")
    if status in {"queued", "generating"}:
        return
    setattr(frame, fields["status_attr"], "queued")
    setattr(frame, fields.get("provider_attr"), "veo")
    save_state(state, get_state_path(job_id))
    print(f"[veo] queued job {job_id} frame {frame_index} variant {variant}")
    thread = threading.Thread(
        target=generate_video_worker,
        args=(job_id, frame_index, variant),
        daemon=True,
    )
    thread.start()


def generate_video_worker(job_id: str, frame_index: int, variant: str) -> None:
    try:
        _generate_veo_video(job_id, frame_index, variant)
    except Exception as exc:
        print(f"[video] generation failed for job {job_id} frame {frame_index} variant {variant}: {exc}")


def _generate_veo_video(job_id: str, frame_index: int, variant: str) -> None:
    if not _veo_enabled():
        raise RuntimeError("Gemini Veo disabled")
    state = ensure_state(job_id)
    frame = state.frames[frame_index]
    fields = _variant_fields(variant)
    setattr(frame, fields["status_attr"], "generating")
    save_state(state, get_state_path(job_id))
    print(f"[veo] generating job {job_id} frame {frame_index} variant {variant}")

    still_rel = getattr(frame, fields["still_attr"])
    still_path = BASE_DIR / Path(still_rel)
    if not still_path.exists():
        raise FileNotFoundError(f"Still image missing for {variant}")

    start_time = time.monotonic()
    try:
        client = _veo_client()
        mime_type, _ = mimetypes.guess_type(str(still_path))
        image_ref = _veo_image_part(still_path, mime_type or "image/png")
        print(f"[veo] uploaded still for job {job_id} frame {frame_index} variant {variant}")
        prompt = _video_prompt(getattr(frame, fields["prompt_attr"]))
        config = _veo_config()
        operation = client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt,
            image=image_ref,
            config=config,
        )
        op_name = getattr(operation, "name", None)
        print(f"[veo] operation {op_name} started for job {job_id} frame {frame_index} variant {variant}")
        state = ensure_state(job_id)
        frame = state.frames[frame_index]
        setattr(frame, fields["job_attr"], op_name)
        save_state(state, get_state_path(job_id))

        poll_operation = operation
        start_time = time.monotonic()
        while not getattr(poll_operation, "done", False):
            elapsed = time.monotonic() - start_time
            if elapsed > VEO_MAX_WAIT_SECONDS:
                raise TimeoutError("Timed out waiting for Veo video")
            print(f"[veo] polling {op_name} (elapsed {elapsed:.0f}s)")
            time.sleep(VEO_POLL_INTERVAL)
            poll_operation = client.operations.get(poll_operation)

        response_payload = getattr(poll_operation, "response", None)
        if response_payload is None:
            raise RuntimeError(f"Veo operation {op_name} completed without response payload: {poll_operation}")
        generated_videos = getattr(response_payload, "generated_videos", [])
        if not generated_videos:
            raise RuntimeError(f"Veo returned no videos (op={op_name}, response={response_payload})")
        video_obj = generated_videos[0].video
        videos_dir = _videos_dir(job_id)
        out_path = videos_dir / f"frame_{frame.index+1:02d}_{variant}.mp4"
        _download_veo_video(client, video_obj, out_path)

        state = ensure_state(job_id)
        frame = state.frames[frame_index]
        setattr(frame, fields["video_attr"], str(relative_to_outputs(out_path)))
        setattr(frame, fields["status_attr"], "ready")
        setattr(frame, fields.get("provider_attr"), "veo")
        save_state(state, get_state_path(job_id))
        elapsed = time.monotonic() - start_time
        print(f"[veo] job {job_id} frame {frame_index} {variant} completed in {elapsed:.1f}s")
    except Exception:
        state = ensure_state(job_id)
        frame = state.frames[frame_index]
        setattr(frame, fields["status_attr"], "failed")
        setattr(frame, fields["job_attr"], None)
        setattr(frame, fields.get("provider_attr"), None)
        save_state(state, get_state_path(job_id))
        print(f"[veo] job {job_id} frame {frame_index} {variant} failed")
        raise


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
    state.frames[frame_index].constraints_version = state.constraints_version

    for idx in range(frame_index + 1, state.num_frames):
        fr = state.frames[idx]
        prompts = build_prompts(fr.caption, state.global_constraints)
        fr.prompt_base, fr.prompt_altA, fr.prompt_altB = prompts
        if idx <= frame_index + LOOKAHEAD_WINDOW:
            _regenerate_frame_assets(state, idx, job_dir)
        else:
            fr.constraints_version = state.constraints_version - 1

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
    _queue_lookahead_videos(job_id, state, frame_index)
    return state


def describe_constraints(frame_index: int, choice: str) -> str:
    frame_num = frame_index + 1
    if choice == "altA":
        return f"Constraints: lock lighting cues from frame {frame_num} onward"
    if choice == "altB":
        return f"Constraints: adopt tighter coverage from frame {frame_num} onward"
    return f"Constraints: preserve baseline look after frame {frame_num}"


def regen_frame(job_id: str, frame_index: int) -> JobState:
    state = ensure_state(job_id)
    if frame_index < 0 or frame_index >= state.num_frames:
        raise ValueError("Frame index out of range")
    frame = state.frames[frame_index]
    if frame.constraints_version >= state.constraints_version:
        return state
    _regenerate_frame_assets(state, frame_index, get_job_dir(job_id))
    save_state(state, get_state_path(job_id))
    _queue_video_if_needed(job_id, frame_index, "base", state)
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
        still_path = BASE_DIR / Path(_chosen_still_path(frame))
        if still_path.exists():
            c.drawImage(
                ImageReader(str(still_path)),
                x,
                y + 0.4 * inch,
                tile_width - 0.2 * inch,
                tile_height - inch,
                preserveAspectRatio=True,
            )
        caption = f"Frame {frame.index+1} choice: {frame.chosen}"
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y + 0.2 * inch, caption)
        c.setFont("Helvetica", 10)
        c.drawString(x, y, frame.caption[:100])
    c.showPage()
    c.save()
    return Path(f"outputs/{job_id}/contact_sheet.pdf")


def _chosen_still_path(frame: Frame) -> str:
    if frame.chosen == "altA":
        return frame.still_altA_path
    if frame.chosen == "altB":
        return frame.still_altB_path
    return frame.still_base_path


def _render_clips_from_stills(frame_number: int, stills: Dict[str, Path], clips_dir: Path) -> Dict[str, str]:
    clips = {}
    for key, label in [("base", "BASE"), ("altA", "ALT A (Lighting)"), ("altB", "ALT B (Camera)")]:
        clip_target = clips_dir / f"frame_{frame_number:02d}_{key}.mp4"
        overlay = f"Beat {frame_number} • {label}"
        still_path = stills[key]
        if render_preview_clip(still_path, clip_target, overlay):
            clips[key] = str(relative_to_outputs(clip_target))
        else:
            clips[key] = str(relative_to_outputs(still_path))
    return clips


def _regenerate_frame_assets(state: JobState, frame_index: int, job_dir: Path) -> None:
    frame = state.frames[frame_index]
    frame_num = frame.index + 1
    enhanced = _ensure_enhanced_path(frame, job_dir)
    stills_dir = job_dir / STILLS_NAME
    lighting_caption = frame.caption_altA or _variant_caption(frame.caption, LIGHTING_CAPTION_SUFFIX)
    camera_caption = frame.caption_altB or _variant_caption(frame.caption, CAMERA_CAPTION_SUFFIX)
    stills = make_variant_stills(
        enhanced,
        stills_dir,
        frame_num,
        frame.caption,
        lighting_caption,
        camera_caption,
        state.global_constraints,
    )
    frame.still_base_path = str(relative_to_outputs(stills["base"]))
    frame.still_altA_path = str(relative_to_outputs(stills["altA"]))
    frame.still_altB_path = str(relative_to_outputs(stills["altB"]))
    frame.caption_altA = lighting_caption
    frame.caption_altB = camera_caption

    clips_dir = job_dir / CLIPS_NAME
    clip_paths = _render_clips_from_stills(frame_num, stills, clips_dir)
    frame.base_clip_path = clip_paths["base"]
    frame.altA_clip_path = clip_paths["altA"]
    frame.altB_clip_path = clip_paths["altB"]
    frame.constraints_version = state.constraints_version
    _invalidate_videos(frame, ("base", "altA", "altB"))


def _ensure_enhanced_path(frame: Frame, job_dir: Path) -> Path:
    enhanced_path = BASE_DIR / Path(frame.enhanced_path)
    if enhanced_path.exists():
        return enhanced_path
    upload = BASE_DIR / Path(frame.upload_path)
    enhanced_dir = job_dir / ENHANCED_NAME
    enhanced = enhance_frame(upload, enhanced_dir / f"frame_{frame.index+1:02d}_enhanced.png")
    frame.enhanced_path = str(relative_to_outputs(enhanced))
    return enhanced


def relative_to_outputs(path: Path) -> Path:
    return Path("outputs") / path.relative_to(OUTPUT_DIR)


def _clamp(text: str, limit: int = 220) -> str:
    return textwrap.shorten(text, width=limit, placeholder="...")


def _veo_config():
    if genai_types is None:
        return None
    return genai_types.GenerateVideosConfig(
        aspect_ratio=VEO_ASPECT_RATIO,
        resolution=VEO_RESOLUTION,
    )


def _download_veo_video(client, video_obj, out_path: Path) -> None:
    """Download a Veo-generated video to disk."""
    download = client.files.download(file=video_obj)
    if hasattr(video_obj, "save"):
        video_obj.save(str(out_path))
        return
    if hasattr(download, "write_to_file"):
        download.write_to_file(str(out_path))
        return
    data_reader = getattr(download, "read", None)
    if callable(data_reader):
        with out_path.open("wb") as outfile:
            outfile.write(data_reader())
        return
    raise RuntimeError("Unable to download Veo video")


def _veo_image_part(still_path: Path, mime_type: str):
    with still_path.open("rb") as img_file:
        data = img_file.read()
    if genai_types is not None and hasattr(genai_types, "Part"):
        part = genai_types.Part.from_bytes(data=data, mime_type=mime_type)
        if hasattr(part, "as_image"):
            return part.as_image()
        return part
    encoded = base64.b64encode(data).decode("utf-8")
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": encoded,
        }
    }


def _safe_overlay_text(text: str) -> str:
    sanitized = text.replace("\n", " ").replace(":", "-").replace("'", "\'")
    sanitized = sanitized.encode("ascii", errors="ignore").decode("ascii")
    return sanitized[:96]


def _sanitize_caption(text: str) -> str:
    clean = text.replace('\n', ' ').replace('—', '-').replace('–', '-').strip()
    return clean


def _queue_lookahead_videos(job_id: str, state: JobState, frame_index: int) -> None:
    if not _veo_enabled():
        return
    for idx in range(frame_index + 1, min(state.num_frames, frame_index + LOOKAHEAD_WINDOW + 1)):
        _queue_video_if_needed(job_id, idx, "base", state)


def _queue_video_if_needed(job_id: str, frame_index: int, variant: str, state: Optional[JobState] = None) -> None:
    if not _veo_enabled():
        return
    if state is None:
        try:
            state = ensure_state(job_id)
        except FileNotFoundError:
            return
    frame = state.frames[frame_index]
    fields = _variant_fields(variant)
    status = getattr(frame, fields["status_attr"], "missing")
    if status in {"ready", "queued", "generating"}:
        return
    generate_video_async(job_id, frame_index, variant)


OUTPUT_DIR.mkdir(exist_ok=True)
