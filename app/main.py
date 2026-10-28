"""FastAPI entrypoint

รัน:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  - Swagger UI       : http://localhost:8000/docs
  - หน้าเว็บมือถือ    : http://localhost:8000/   (ถ่ายรูปจากกล้องมือถือ)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import router
from app.config import settings
from app.db import init_db
from app.review.service import ReviewError

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(
    title="Intelligent Expiration Date AI OCR System with Human-in-the-Loop Self-Correction",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.exception_handler(ReviewError)
async def review_error_handler(_: Request, e: ReviewError):
    return JSONResponse(status_code=e.status_code, content={"detail": {"message": str(e), "details": e.details}})


@app.get("/health")
def health():
    return {"status": "ok", "engine": settings.ocr_engine, "threshold": settings.confidence_threshold,
            "ensemble_passes": settings.ensemble_passes}


@app.get("/", include_in_schema=False)
def mobile_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return FileResponse(WEB_DIR / "manifest.webmanifest", media_type="application/manifest+json")
