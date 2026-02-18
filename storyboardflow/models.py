"""Data models for storyboard branching workflow."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Variant(BaseModel):
    key: str
    kind: str = "base"
    parent_key: Optional[str] = None
    still_path: str
    preview_clip_path: Optional[str] = None
    ai_video_status: str = "missing"
    ai_video_path: Optional[str] = None
    ai_job_id: Optional[str] = None


class Frame(BaseModel):
    index: int
    upload_path: str
    enhanced_path: str = ""
    still_base_path: str = ""
    still_altA_path: str = ""
    still_altB_path: str = ""
    caption: str
    caption_altA: str = ""
    caption_altB: str = ""
    prompt_base: str
    prompt_altA: str
    prompt_altB: str
    base_clip_path: str
    altA_clip_path: str
    altB_clip_path: str
    video_base_path: Optional[str] = None
    video_altA_path: Optional[str] = None
    video_altB_path: Optional[str] = None
    video_base_status: str = "missing"
    video_altA_status: str = "missing"
    video_altB_status: str = "missing"
    video_base_job_id: Optional[str] = None
    video_altA_job_id: Optional[str] = None
    video_altB_job_id: Optional[str] = None
    video_base_provider: Optional[str] = None
    video_altA_provider: Optional[str] = None
    video_altB_provider: Optional[str] = None
    chosen: str = "base"
    constraints_version: int = 0
    is_stale: bool = False
    variants: Dict[str, Variant] = Field(default_factory=dict)
    chosen_variant_key: str = "base"


class JobState(BaseModel):
    job_id: str
    num_frames: int
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    constraints_version: int = 0
    global_constraints: str = ""
    active_constraints: dict = Field(default_factory=lambda: {
        "lighting": None,
        "camera": None,
        "mood": None,
        "style": None,
    })
    constraints_history: List[dict] = Field(default_factory=list)
    global_mode: Optional[str] = None
    global_mode_start_index: Optional[int] = None
    frames: List[Frame]
    branch_events: List[dict] = Field(default_factory=list)

    def compute_staleness(self) -> None:
        for frame in self.frames:
            frame.is_stale = frame.constraints_version < self.constraints_version

    def frame_by_index(self, idx: int) -> Frame:
        return self.frames[idx]


def load_state(path: Path) -> Optional[JobState]:
    if not path.exists():
        return None
    text = path.read_text()
    if not text:
        return None
    state = JobState.model_validate_json(text)
    for frame in state.frames:
        if frame.variants:
            continue
        variants: Dict[str, Variant] = {}
        # Base variant from existing still
        variants["base"] = Variant(
            key="base",
            kind="base",
            still_path=frame.still_base_path or frame.enhanced_path or frame.upload_path,
            preview_clip_path=frame.base_clip_path,
            ai_video_status=frame.video_base_status,
            ai_video_path=frame.video_base_path,
            ai_job_id=frame.video_base_job_id,
        )
        if frame.still_altA_path:
            variants["alt-lighting"] = Variant(
                key="alt-lighting",
                kind="alt-lighting",
                still_path=frame.still_altA_path,
                preview_clip_path=frame.altA_clip_path,
                ai_video_status=frame.video_altA_status,
                ai_video_path=frame.video_altA_path,
                ai_job_id=frame.video_altA_job_id,
            )
        if frame.still_altB_path:
            variants["alt-camera"] = Variant(
                key="alt-camera",
                kind="alt-camera",
                still_path=frame.still_altB_path,
                preview_clip_path=frame.altB_clip_path,
                ai_video_status=frame.video_altB_status,
                ai_video_path=frame.video_altB_path,
                ai_job_id=frame.video_altB_job_id,
            )
        frame.variants = variants
        if not frame.chosen_variant_key:
            frame.chosen_variant_key = "base"
    state.compute_staleness()
    return state


def save_state(state: JobState, path: Path) -> None:
    state.compute_staleness()
    path.write_text(state.model_dump_json(indent=2))
