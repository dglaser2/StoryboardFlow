"""Data models for storyboard branching workflow."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


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
    chosen: str = "base"
    constraints_version: int = 0
    is_stale: bool = False


class JobState(BaseModel):
    job_id: str
    num_frames: int
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    constraints_version: int = 0
    global_constraints: str = ""
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
    state.compute_staleness()
    return state


def save_state(state: JobState, path: Path) -> None:
    state.compute_staleness()
    path.write_text(state.model_dump_json(indent=2))
