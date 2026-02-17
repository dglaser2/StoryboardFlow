"""Pydantic models for storyboardflow pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class Scene(BaseModel):
    title: str
    characters: List[str]
    location: str
    time_of_day: str
    mood: str
    lighting: str
    camera: str
    style: str
    constraints: List[str] = Field(default_factory=list)


class Variation(BaseModel):
    name: str
    prompt: str
    image_path: str
