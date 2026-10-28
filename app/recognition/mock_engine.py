"""Mock OCR engine — ใช้รันระบบ/สาธิต/ทดสอบได้โดยไม่ต้องมี GPU หรือโหลดโมเดล

หลักการ: ภาพสังเคราะห์จาก scripts/generate_synthetic.py ฝัง ground truth ไว้ใน PNG metadata
(key = "ocr_gt") engine นี้อ่านค่านั้นแล้ว "จำลองความผิดพลาดของ OCR" อย่างสมจริง:

- อัตราผิดพลาดขึ้นกับคุณภาพภาพที่วัดได้จริง (ความเบลอ, คอนทราสต์) ของภาพที่ส่งเข้ามา
  => preprocessing ที่ดีทำให้ผลดีขึ้นจริง
- ตัวอักษรที่สับสนบ่อย (0/O, 1/I/l, 5/S, 8/B, 2/Z, 6/G) จะถูกสลับพร้อม confidence ต่ำ
- มีโอกาสเล็กน้อยเกิด "confident error" (ผิดแต่มั่นใจสูง) เพื่อให้เห็นว่ากฎธุรกิจ + ensemble จำเป็น
- ความผิดพลาดส่วนหนึ่ง "ติดมากับภาพ" (ทุกรอบผิดเหมือนกัน) อีกส่วนสุ่มตามรอบ (pass_index)
  => จำลองการอ่านหลายรอบ (multi-pass ensemble) ได้สมจริง ไม่ใช่ว่า ensemble จับได้ทุกเคส

ภาพที่ไม่มี metadata (เช่น ภาพถ่ายจริง) จะคืนผลว่าง => ระบบส่งเข้า HITL ทั้งหมด
ใช้ OCR_ENGINE=trocr หรือ vlm สำหรับภาพจริง
"""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np
from PIL import Image

from app.intake.preprocessing import blur_score, contrast_score, transform_points
from app.recognition.base import OCREngine, RecognitionContext, TextLine

CONFUSIONS = {
    "0": "O", "O": "0", "1": "I", "I": "1", "l": "1", "5": "S", "S": "5", "8": "B", "B": "8",
    "2": "Z", "Z": "2", "6": "G", "G": "6", "3": "8", "7": "1", "/": "1", ".": ",", "9": "g",
}


def read_ground_truth(raw_bytes: bytes) -> dict | None:
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        meta = getattr(img, "text", {}) or img.info
        if "ocr_gt" in meta:
            return json.loads(meta["ocr_gt"])
    except Exception:
        return None
    return None


class MockOCREngine(OCREngine):
    name = "mock"

    def __init__(self, base_error: float = 0.003, confident_error_rate: float = 0.08, shared_ratio: float = 0.6):
        self.base_error = base_error
        self.confident_error_rate = confident_error_rate
        self.shared_ratio = shared_ratio   # สัดส่วนความผิดพลาดที่ "ติดมากับภาพ" (ทุกรอบผิดเหมือนกัน)

    def _error_rate(self, image: np.ndarray) -> float:
        blur = blur_score(image)
        contrast = contrast_score(image)
        rate = self.base_error
        rate += 0.08 * float(np.clip((150 - blur) / 150, 0, 1))
        rate += 0.04 * float(np.clip((20 - contrast) / 20, 0, 1))
        return float(min(rate, 0.35))

    def recognize(self, image: np.ndarray, ctx: RecognitionContext) -> list[TextLine]:
        gt = read_ground_truth(ctx.raw_bytes)
        if not gt:
            return []
        # rng_img: ความผิดพลาดที่เกิดจากตัวภาพเอง (เหมือนกันทุกรอบ) / rng_pass: เฉพาะรอบนั้น
        def seed(tag):
            return int(hashlib.sha256(f"{ctx.sha256}:{tag}".encode()).hexdigest()[:8], 16)
        rng_img = np.random.default_rng(seed("image"))
        rng = np.random.default_rng(seed(ctx.pass_index))
        err = self._error_rate(image)
        lines = []
        for line in gt.get("lines", []):
            chars, confs = [], []
            for ch in line["text"]:
                shared_hit = rng_img.random() < err * self.shared_ratio
                shared_confident = rng_img.random() < self.confident_error_rate
                pass_hit = rng.random() < err * (1 - self.shared_ratio)
                if ch in CONFUSIONS and (shared_hit or pass_hit):
                    chars.append(CONFUSIONS[ch])
                    confident = shared_confident if shared_hit else rng.random() < self.confident_error_rate
                    confs.append(float(rng.uniform(0.91, 0.97)) if confident else float(rng.uniform(0.35, 0.85)))
                else:
                    chars.append(ch)
                    confs.append(float(rng.uniform(max(0.9, 0.995 - err), 0.999)))
            poly = line.get("polygon")
            if poly is not None and ctx.matrix is not None:
                poly = transform_points(poly, ctx.matrix).round(1).tolist()
            text = "".join(chars)
            lines.append(TextLine(text=text, confidence=float(np.mean(confs)) if confs else 0.0,
                                  char_confidences=confs, polygon=poly))
        return lines
