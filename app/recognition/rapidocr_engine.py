"""RapidOCR engine — AI OCR ตัวจริงที่รันบน CPU ได้เร็ว (ไม่ต้องมี GPU / API key)

ใช้โมเดล PP-OCR (PaddleOCR) แปลงเป็น ONNX: Text Detection (DBNet) + Text Recognition (SVTR/CRNN)
- ติดตั้ง: pip install rapidocr-onnxruntime   (โมเดลติดมากับแพ็กเกจ ไม่ต้องดาวน์โหลดเพิ่ม)
- ผลลัพธ์ระดับ "คำ" จะถูกรวมเป็น "บรรทัด" ตามตำแหน่งแนวตั้ง เพื่อให้ KIE หา label + วันที่ในบรรทัดเดียวกันได้
- ความมั่นใจ: RapidOCR ให้คะแนนระดับคำ (ค่าเฉลี่ย probability ของตัวอักษร) จึงกระจายให้ทุกตัวอักษรในคำ
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.recognition.base import OCREngine, RecognitionContext, TextLine


def group_words_into_lines(words: list[tuple[list, str, float]], overlap: float = 0.5) -> list[TextLine]:
    """รวมกล่องคำที่อยู่แนวเดียวกันเป็นบรรทัด (เรียงบนลงล่าง ซ้ายไปขวา)"""
    items = []
    for box, text, score in words:
        pts = np.asarray(box, dtype=float)
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        items.append({"pts": pts, "text": str(text).strip(), "score": float(score),
                      "y0": y0, "y1": y1, "cy": (y0 + y1) / 2, "x0": pts[:, 0].min()})
    items = [i for i in items if i["text"]]
    # เรียงซ้าย->ขวา แล้วต่อคำเข้ากับบรรทัดที่ "คำขวาสุด" อยู่แนวเดียวกัน (รองรับข้อความเอียงเล็กน้อย)
    items.sort(key=lambda i: i["x0"])
    lines: list[list[dict]] = []
    for it in items:
        best, best_score = None, 0.0
        for line in lines:
            last = line[-1]
            inter = min(last["y1"], it["y1"]) - max(last["y0"], it["y0"])
            score = inter / max(1e-6, min(last["y1"] - last["y0"], it["y1"] - it["y0"]))
            if score > overlap and score > best_score:
                best, best_score = line, score
        if best is None:
            lines.append([it])
        else:
            best.append(it)
    out = []
    for line in lines:
        line.sort(key=lambda w: w["x0"])
        text, confs = "", []
        for k, w in enumerate(line):
            if k:
                text += " "
                confs.append(min(w["score"], line[k - 1]["score"]))
            text += w["text"]
            confs.extend([w["score"]] * len(w["text"]))
        pts = np.vstack([w["pts"] for w in line])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        poly = [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]
        out.append(TextLine(text=text, confidence=float(np.mean(confs)), char_confidences=confs, polygon=poly))
    out.sort(key=lambda ln: ((ln.polygon or [[0, 0]])[0][1], (ln.polygon or [[0, 0]])[0][0]))
    return out


class RapidOCREngine(OCREngine):
    name = "rapidocr"

    def __init__(self) -> None:
        self._ocr: Any = None

    def _load(self) -> Any:
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # pyright: ignore[reportMissingImports]
            except ImportError as e:
                raise RuntimeError("ต้องติดตั้งก่อน: pip install rapidocr-onnxruntime") from e
            self._ocr = RapidOCR()
        return self._ocr

    def recognize(self, image: np.ndarray, ctx: RecognitionContext) -> list[TextLine]:
        result, _ = self._load()(image)
        if not result:
            return []
        return group_words_into_lines([(box, text, float(score)) for box, text, score in result])
