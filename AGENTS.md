Repository Guidelines

Project Overview

StoryboardFlow is a minimal Python-only FastAPI application that demonstrates a “creative variation engine”:

Text scene → Structured Scene Interpretation → 4 controlled visual variations → Contact-sheet PDF export.

The app must always run end-to-end without API keys using deterministic fallbacks.

⸻

Project Structure & Module Organization
app.py              # FastAPI app + routing
pipeline.py         # Core scene extraction + variation logic
models.py           # Pydantic models (Scene, Variation)
prompts.py          # LLM prompt templates (optional)
templates/          # Jinja2 HTML templates
outputs/            # Generated assets (gitignored)
requirements.txt
README.md

	•	Business logic belongs in pipeline.py.
	•	API and routing belong in app.py.
	•	No additional services, databases, or background jobs.

⸻

Build, Run & Development Commands

Create virtual environment and install:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run locally:
uvicorn app:app --reload

Outputs are written to:
outputs/<job_id>/

Do not require FFmpeg or API keys for core functionality.

⸻

Coding Style & Naming Conventions
	•	Python 3.11+
	•	4-space indentation
	•	Use Pydantic models for structured data
	•	Functions should be small and deterministic
	•	Prefer explicit naming (e.g., extract_scene, make_variation_prompts)
	•	Avoid over-engineering or adding new dependencies

This project values clarity over abstraction.

⸻

Testing Guidelines

No formal test suite required for this demo.

However:
	•	All core logic must be callable independently from pipeline.py
	•	The app must run without external credentials
	•	The UI must render even if optional features fail

Graceful degradation is mandatory.

⸻

Commit & Pull Request Guidelines

Commit messages should be concise and imperative:
Add variation prompt generator
Fix static mounting for outputs
Improve placeholder image readability

Pull requests must:
	•	Keep scope small
	•	Preserve fallback behavior
	•	Avoid adding infrastructure (DB, auth, async workers)

⸻

Architecture Constraints (Critical)
	•	Single-scene input only
	•	Exactly 4 variations:
	•	Faithful
	•	Lighting
	•	Camera
	•	Emotion
	•	No persistence layer
	•	No authentication
	•	No background task queues
	•	Must function without any external API keys

⸻

Agent-Specific Instructions (Codex)
	•	Do not expand scope beyond Variation Mode.
	•	Do not introduce new frameworks.
	•	Do not convert to SPA / React.
	•	Keep HTML simple and server-rendered.
	•	Ensure static /outputs mounting works.
	•	Always preserve deterministic fallback logic.

When unsure, choose the simplest implementation that satisfies the spec.
