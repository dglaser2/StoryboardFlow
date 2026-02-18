"""FastAPI routes for storyboard branching workflow."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .pipeline import (
    OUTPUT_DIR,
    choose_variant,
    create_job,
    ensure_state,
    export_pdf,
    generate_video_async,
    regen_frame,
    queue_ai_clip,
)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="StoryboardFlow — Rough Board")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/create")
async def create_storyboard(request: Request, files: List[UploadFile] = File(...)):
    items: List[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        if not content:
            continue
        filename = upload.filename or f"frame_{len(items)+1}.png"
        items.append((filename, content))
    try:
        result = create_job(items)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(result.redirect_url, status_code=303)


@app.get("/review/{job_id}", response_class=HTMLResponse)
async def review(request: Request, job_id: str):
    state = ensure_state(job_id)
    frames = [frame.model_dump() for frame in state.frames]
    pdf_path = Path(f"outputs/{job_id}/contact_sheet.pdf")
    context = {
        "request": request,
        "state": state,
        "frames": frames,
        "job_id": job_id,
        "pdf_path": str(pdf_path) if pdf_path.exists() else None,
        "hq_enabled": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    }
    return templates.TemplateResponse("review.html", context)


@app.post("/choose/{job_id}")
async def choose(job_id: str, frame_index: int = Form(...), choice: str = Form(...)):
    try:
        choose_variant(job_id, frame_index, choice)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    anchor = f"#frame-{frame_index + 1}"
    return RedirectResponse(url=f"/review/{job_id}{anchor}", status_code=303)


@app.post("/regen/{job_id}")
async def regen(request: Request, job_id: str, frame_index: int = Form(...)):
    try:
        regen_frame(job_id, frame_index)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"status": "ok"})
    anchor = f"#frame-{frame_index + 1}"
    return RedirectResponse(url=f"/review/{job_id}{anchor}", status_code=303)


@app.post("/generate_video/{job_id}")
async def generate_video(job_id: str, frame_index: int = Form(...), variant: str = Form(...)):
    try:
        generate_video_async(job_id, frame_index, variant)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "queued"})


@app.get("/api/job/{job_id}")
async def job_state(job_id: str):
    state = ensure_state(job_id)
    return {
        "frames": [frame.model_dump() for frame in state.frames],
        "constraints_version": state.constraints_version,
    }


@app.post("/generate_ai_clip/{job_id}")
async def generate_ai_clip(job_id: str, frame_index: int = Form(...), variant_key: str = Form(...)):
    try:
        queue_ai_clip(job_id, frame_index, variant_key)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "queued"})


@app.post("/export/{job_id}")
async def export(job_id: str):
    try:
        export_pdf(job_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=f"/review/{job_id}", status_code=303)


@app.get("/health", response_model=dict)
async def healthcheck():
    return {"status": "ok"}
