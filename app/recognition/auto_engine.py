"""Auto engine (ค่าเริ่มต้น)

- ภาพสังเคราะห์ที่มี ground truth ฝังอยู่ (จาก scripts/generate_synthetic.py) -> MockOCREngine (สาธิต/ทดสอบ)
- ภาพจริงทุกภาพ (เช่น ถ่ายจากมือถือ) -> RapidOCR (AI OCR ตัวจริง)
"""
from __future__ import annotations

import numpy as np

from app.recognition.base import OCREngine, RecognitionContext, TextLine
from app.recognition.mock_engine import MockOCREngine, read_ground_truth
from app.recognition.rapidocr_engine import RapidOCREngine


class AutoEngine(OCREngine):
    name = "auto"

    def __init__(self, real: OCREngine | None = None) -> None:
        self.mock = MockOCREngine()
        self.real = real or RapidOCREngine()
        self.last_used = self.real.name

    def recognize(self, image: np.ndarray, ctx: RecognitionContext) -> list[TextLine]:
        engine = self.mock if read_ground_truth(ctx.raw_bytes) else self.real
        self.last_used = engine.name
        return engine.recognize(image, ctx)
