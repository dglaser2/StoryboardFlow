# storyboardflow

StoryboardFlow is a minimal “variation mode” engine that turns a single scene description into a structured interpretation, four creative image tiles, and a downloadable contact-sheet PDF. Everything runs locally with deterministic fallbacks so you can explore creative branches without API keys.

## 30-second local run
1. Install deps (Python 3.11+):
   ```bash
   cd storyboardflow
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn storyboardflow.app:app --reload
   ```
2. Visit http://127.0.0.1:8000, paste a scene, and submit.

## Example scene to paste
> A drenched detective waits beneath a flickering streetlamp as distant sirens cut through the midnight rain.

## What variation mode demonstrates
Variation mode keeps core story beats locked while intentionally perturbing lighting, camera, and emotional tone. The faithfulness tile grounds the scene, and the remaining three tiles show controlled creative iteration for lighting, camera, and emotion choices. The PDF contact sheet captures all four frames with captions for quick reviews.

## Optional: enable OpenAI
Set `OPENAI_API_KEY` in your shell before launching Uvicorn to let the pipeline call OpenAI for scene parsing and prompt drafting:
```bash
export OPENAI_API_KEY=sk-...
uvicorn storyboardflow.app:app --reload
```
Without the key, the heuristics infer scene fields, create prompts, render Pillow placeholders, and generate the PDF.
