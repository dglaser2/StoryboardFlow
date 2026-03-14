# StoryboardFlow

StoryboardFlow is a minimal FastAPI playground for a creative variation engine: feed it a single storyboard scene, let the pipeline interpret the beat, and instantly explore four deterministic visual viewpoints before exporting a polished contact sheet. Everything runs locally with pure Python dependencies, no API keys required, and every optional integration (captions, Ken Burns clips, Gemini Veo renders) gracefully falls back to deterministic placeholders.

> Text scene ➜ Structured scene interpretation ➜ Faithful + Lighting + Camera + Emotion variations ➜ Contact-sheet PDF export.

## Demo

GitHub strips inline `<video>` tags from README files, so use the direct link below to watch the clip:

- [▶️ Watch the 55-second demo (MP4)](storyboardflow/docs/storyboardflow-demo.mp4)

## Why StoryboardFlow?
- **Deterministic by design** – no external services are required; captions, prompts, and image treatments all fall back to reproducible heuristics so every run can be recorded.
- **Exactly four creative lanes** – every scene surfaces Faithful, Lighting, Camera, and Emotion interpretations so creative leads can compare intent versus stylization without branching into bespoke prompts.
- **Review-first UI** – the FastAPI/Jinja UI keeps the loop tight: upload ➜ branch ➜ regenerate ➜ export, with inline regen buttons, stale indicators, and instant previews.
- **Outputs you can ship** – each job writes annotated stills, preview clips, and a landscape PDF contact sheet to `storyboardflow/outputs/<job_id>/` so you can drop results straight into a deck.

## How it works
1. **Upload a single scene** – paste a scene description (or upload reference stills) via `/`; the backend stores assets under `storyboardflow/outputs/<job_id>/uploads/`.
2. **Scene interpretation** – `storyboardflow/pipeline.py` captions the scene via OpenAI Vision *if* `OPENAI_API_KEY` is set, otherwise a deterministic heuristic emits `Storyboard frame X…` placeholders.
3. **Variation fan-out** – `make_variant_stills` enhances the base frame and renders three direct variants plus an Emotion overlay using only Pillow.
4. **Branching review** – `/review/{job_id}` surfaces each variation with hover clips; picking a lane bumps the constraints version, regenerates the next two frames (lookahead window = 2), and queues optional AI clips.
5. **Contact sheet export** – `/export/{job_id}` runs `export_pdf`, dropping `storyboardflow/outputs/<job_id>/contact_sheet.pdf` so stakeholders can compare lanes offline.

## Variation lanes

| Lane | Purpose | How it is produced |
| --- | --- | --- |
| **Faithful** | Keep the scene as interpreted; sanity check the structured beat. | Enhanced still + caption overlay via Pillow. |
| **Lighting** | Stress-test the lighting direction without touching camera blocking. | `_apply_lighting_grade` ramps color, contrast, and vignettes per deterministic constraints. |
| **Camera** | Explore coverage changes (push/pull, OTS, low angle). | `_apply_camera_variant` crops, pans, and sharpens for cinematic framing. |
| **Emotion** | Surface an alternate mood to test story tone. | Derived overlays are created when you branch; `_create_*_derivatives` spawns mood-forward stills so you can compare to the canonical lanes. |

Each lane preserves caption + prompt context so optional downstream generators (Ken Burns preview clips, Gemini Veo video renders) inherit consistent instructions.

## Deterministic fallbacks
- **Captions** – without `OPENAI_API_KEY`, captions default to `Storyboard frame N…` to keep prompts stable.
- **Preview clips** – `render_preview_clip` only runs if `ffmpeg` is on `PATH`; otherwise the UI automatically displays the still.
- **AI video buttons** – the "Generate AI Clip" CTA hides itself unless `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present *and* `google-genai` is installed.
- **Derived variants** – even if an AI generation call fails, base/lane PNGs live under `storyboardflow/outputs/` so you never lose work.

## Quickstart

```bash
cd storyboardflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Visit http://127.0.0.1:8000 to submit your scene, then follow the redirect to `/review/<job_id>`.

### Optional integrations
- **OpenAI captions** – set `OPENAI_API_KEY` to unlock GPT-4o vision captions (fallback text remains available).
- **Gemini Veo clips** – export `GEMINI_API_KEY` or `GOOGLE_API_KEY` and keep `google-genai` installed to allow `/generate_ai_clip` jobs; results land inside `storyboardflow/outputs/<job_id>/ai_videos/`.
- **ffmpeg previews** – install ffmpeg (`brew install ffmpeg`) for Ken Burns-style MP4 previews; otherwise GIF-like stills render instantly.

## Project layout

```
storyboardflow/app.py          # FastAPI routes + static mounting
storyboardflow/pipeline.py     # Scene analysis + variation/regen/export logic
storyboardflow/models.py       # Pydantic Frame/Variant/JobState definitions
storyboardflow/prompts.py      # Prompt helpers (optional)
storyboardflow/templates/      # index/review Jinja templates
storyboardflow/outputs/        # Generated assets (gitignored)
storyboardflow/docs/           # README-facing assets (demo video)
```

All business logic lives in `storyboardflow/pipeline.py`, so you can call `pipeline.create_job(...)`, `pipeline.choose_variant(...)`, or `pipeline.export_pdf(...)` from a REPL to test steps without the web UI.

## Outputs

A successful run creates a deterministic folder layout:

```
storyboardflow/outputs/<job_id>/
├── uploads/           # raw submission
├── enhanced/          # Pillow-enhanced stills
├── stills/            # Faithful/Lighting/Camera/Emotion PNGs
├── clips/             # Ken Burns MP4s (falls back to PNGs)
├── videos/            # Gemini Veo renders when enabled
├── ai_videos/         # Async AI clip jobs
└── contact_sheet.pdf  # Landscape PDF export
```

You own everything inside `storyboardflow/outputs/`, so it is safe to drop the folder into Dropbox, Slack, or a pitch deck as-is.

## Development notes
- Routes live in [`storyboardflow/app.py`](storyboardflow/app.py); `/health` returns a simple JSON payload for uptime checks.
- The UI intentionally stays server-rendered (Jinja + vanilla JS) to keep the repo dependency-light.
- No databases, task queues, or background workers are required—the only concurrency lives inside the deterministic thread pool used for optional AI clips.
- Because every stage is deterministic, you can unit-test pieces by importing pipeline functions and passing fixture data without spinning up FastAPI.

Have fun remixing your scenes ✦
