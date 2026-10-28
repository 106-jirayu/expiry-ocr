"""Capture & Recognition Layer — interface กลางของ OCR/VLM engine

ทุก engine คืนผลเป็นรายการ TextLine (ข้อความ 1 บรรทัด + ความมั่นใจระดับตัวอักษร + polygon)
ทำให้สลับ TrOCR / VLM / Mock ได้โดยไม่กระทบชั้นอื่น (Adapter pattern)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TextLine:
    text: str
    confidence: float                         # ความมั่นใจของทั้งบรรทัด (0-1)
    char_confidences: list[float]             # ความมั่นใจรายตัวอักษร (ยาวเท่ากับ text)
    polygon: list[list[float]] | None = None  # [[x,y]*4] ในพิกัดภาพที่ส่งเข้า engine

    def __post_init__(self):
        if len(self.char_confidences) != len(self.text):
            # engine บางตัวให้มาแค่ระดับบรรทัด -> กระจายค่าเดียวกันให้ทุกตัวอักษร
            self.char_confidences = [self.confidence] * len(self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "char_confidences": [round(c, 4) for c in self.char_confidences],
            "polygon": self.polygon,
        }


@dataclass
class RecognitionContext:
    """ข้อมูลประกอบที่ engine อาจใช้ (เช่น ไฟล์ต้นฉบับ, affine matrix, pass index ของ ensemble)"""
    raw_bytes: bytes = b""
    sha256: str = ""
    matrix: np.ndarray | None = None          # original -> processed
    original_image: np.ndarray | None = None
    pass_index: int = 0
    extra: dict = field(default_factory=dict)


class OCREngine(ABC):
    name: str = "base"

    @abstractmethod
    def recognize(self, image: np.ndarray, ctx: RecognitionContext) -> list[TextLine]:
        """อ่านข้อความทั้งหมดในภาพ คืนรายการบรรทัดเรียงจากบนลงล่าง"""
