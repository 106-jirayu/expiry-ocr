from __future__ import annotations

from app.config import Settings, settings
from app.recognition.base import OCREngine


def build_engine(name: str | None = None, cfg: Settings = settings) -> OCREngine:
    name = (name or cfg.ocr_engine).lower()
    if name == "auto":
        from app.recognition.auto_engine import AutoEngine
        return AutoEngine()
    if name == "rapidocr":
        from app.recognition.rapidocr_engine import RapidOCREngine
        return RapidOCREngine()
    if name == "mock":
        from app.recognition.mock_engine import MockOCREngine
        return MockOCREngine()
    if name == "trocr":
        from app.recognition.trocr_engine import TrOCREngine
        return TrOCREngine(cfg.trocr_model)
    if name == "vlm":
        from app.recognition.vlm_engine import VLMEngine
        return VLMEngine(cfg.vlm_endpoint, cfg.vlm_model, cfg.vlm_api_key)
    raise ValueError(f"unknown OCR engine: {name}")
