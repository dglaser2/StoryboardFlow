"""FastAPI application entry point."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .pipeline import OUTPUT_DIR, run_pipeline

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="StoryboardFlow Variation Mode")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, scene_text: str = Form(...)):
    try:
        payload = run_pipeline(scene_text)
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    context = {
        "request": request,
        "scene": payload["scene"],
        "variations": payload["variations"],
        "job_id": payload["job_id"],
        "pdf_path": payload["pdf_path"],
    }
    return templates.TemplateResponse("result.html", context)


@app.get("/health", response_model=dict)
async def healthcheck():
    return {"status": "ok"}
