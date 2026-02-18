# StoryboardFlow — Rough Board ➜ Preview Clips

StoryboardFlow turns an uploaded rough storyboard (4–8 frames) into captioned, gently enhanced review clips with branching alternatives. Each frame is upscaled with a Pillow-only pass, captioned (vision if available), and then fanned out into a base still plus two deterministic variations (lighting grade + camera crop). Choosing an alternative propagates constraints downstream with a lookahead regeneration window of 2 frames so decisions feel immediate, while later frames lazily refresh when you inspect them.

## 30-second local run
```bash
cd storyboardflow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn storyboardflow.app:app --reload
```
Visit http://127.0.0.1:8000, upload 4–8 JPG/PNG frames, and start branching.

## What variation mode demonstrates
- **Captions**: each uploaded frame gets a 1-line caption (OpenAI vision if `OPENAI_API_KEY` is set, deterministic fallback otherwise) which appears directly in the review grid.
- **Upscale + enhancement**: Pillow resizes every frame to 1280px wide and applies a mild contrast/sharpness pass so the derived stills don’t look soft even before alt treatment.
- **Deterministic lighting + camera alts**: Alt A performs a stylized color grade (warm/cool bias + vignette) while Alt B tightens the crop with a subtle pan; both feed into still images and the hover clips.
- **Preview clips**: ffmpeg applies a Ken Burns zoompan and text overlay for Base, Alt A, and Alt B. If ffmpeg is missing, StoryboardFlow falls back to the generated stills automatically.
- **Branching with lookahead regen (W=2)**: selecting an alternative updates global constraints, re-renders the next two frames immediately (stills + clips), and marks later frames stale until you interact with them (lazy regen via `POST /regen`).
- **Contact sheet export**: render the chosen variants into a PDF via ReportLab.
- **Optional AI video generation**: preview clips always render instantly, and you can queue a high-quality OpenAI Sora clip asynchronously per variant. The UI surfaces HQ statuses (missing/queued/generating/ready) and swaps in the real clip automatically when it arrives.

## Example workflow
1. Upload your rough storyboard images and submit.
2. Review the grid — hover to play clips. Stale badges indicate frames waiting for regeneration.
3. Click a tile to open Alternatives. Choose Base, Alt A (lighting), or Alt B (camera). The system bumps the constraint version, re-renders the next two frames, and queues later frames for on-demand regen.
4. If you hit a stale frame later, clicking it triggers quick regeneration before showing options.
5. Export a contact sheet PDF when ready via the Review page.

## Optional: enable OpenAI + ffmpeg
- **OpenAI (for captions)**
  ```bash
  export OPENAI_API_KEY=sk-...
  ```
- **ffmpeg (for animated previews)** – install via Homebrew (`brew install ffmpeg`) or your package manager. Without it, StoryboardFlow automatically falls back to still images and disables hover playback.
  
## Optional: Gemini Veo HQ clips
StoryboardFlow can now hand off HQ clip generation to Google’s Veo 3.1 via the Gemini API (defaulting to 3‑second 720p shots). Preview clips are still created instantly; Veo runs asynchronously, and the UI swaps to the HQ version when the job finishes.

1. Install the Google client (already listed in `requirements.txt`):
   ```bash
   pip install google-genai
   ```
2. Export your Gemini key:
   ```bash
   export GEMINI_API_KEY=AIza...
   ```
3. Start `uvicorn` as usual and click “Generate HQ Clip” on any frame. The status badge shows `queued → generating → ready`, and the MP4 lands in `outputs/<job>/videos/`.

If no Gemini key is present, HQ buttons stay hidden and the system falls back to ffmpeg previews automatically.
