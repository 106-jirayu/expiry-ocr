"""Confidence Calibration — แปลงคะแนนดิบของ OCR ให้เป็น "ความน่าจะเป็นที่อ่านถูกจริง"

ปัญหา: OCR แต่ละตัวให้คะแนนคนละสเกล เช่น RapidOCR อ่านถูกแต่ให้คะแนน 0.75-0.90
ถ้าใช้เกณฑ์ 90% ตรง ๆ ทุกภาพจะเข้าคิวคนตรวจหมด (FPY ≈ 0)

วิธี: เก็บชุดข้อมูลที่มีเฉลย -> รัน OCR -> จับคู่ (คะแนนดิบ, ถูก/ผิด) ของแต่ละฟิลด์
     แบ่งเป็นช่วง (equal-frequency bins) แล้วใช้ "Wilson lower bound" ของความแม่นในแต่ละช่วง
     (ประมาณการแบบระมัดระวัง ข้อมูลน้อย = คะแนนต่ำ) + บังคับให้เพิ่มขึ้นตามคะแนนดิบ (isotonic / PAV)
ผลลัพธ์: คะแนนหลัง calibrate 0.90 = "ในอดีต ฟิลด์ที่ได้คะแนนระดับนี้ถูกอย่างน้อย 90%" (ที่ความเชื่อมั่น 95%)

สร้างไฟล์ calibration: python -m evaluation.evaluate --engine rapidocr --fit-calibration
ควร fit ใหม่ด้วย "ภาพถ่ายจริง" ที่มีเฉลย เพราะการกระจายของคะแนนขึ้นกับลักษณะภาพ
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from app.extraction.kie import FieldPrediction


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / den)


def fit(raw: list[float], correct: list[bool], n_bins: int = 10, min_per_bin: int = 15) -> list[list[float]]:
    """คืนจุด [[คะแนนดิบ, คะแนนหลัง calibrate], ...] เรียงจากน้อยไปมาก"""
    pairs = sorted(zip(raw, correct))
    if not pairs:
        return []
    size = max(min_per_bin, len(pairs) // n_bins)
    bins = [pairs[i:i + size] for i in range(0, len(pairs), size)]
    if len(bins) > 1 and len(bins[-1]) < min_per_bin:
        bins[-2].extend(bins.pop())
    # pool-adjacent-violators บนจำนวนถูก/ทั้งหมด เพื่อให้ความแม่นไม่ลดลงเมื่อคะแนนดิบเพิ่ม
    blocks = [[sum(c for _, c in b), len(b), b[0][0], b[-1][0]] for b in bins]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            k, n, lo, _ = blocks[i]
            k2, n2, _, hi2 = blocks.pop(i + 1)
            blocks[i] = [k + k2, n + n2, lo, hi2]
            i = max(0, i - 1)
        else:
            i += 1
    points, last = [], 0.0
    for k, n, lo, hi in blocks:
        cal = max(last, wilson_lower(k, n))
        points.append([round(lo, 4), round(cal, 4)])
        last = cal
    return points


def apply_points(points: list[list[float]], raw: float) -> float:
    """step function: ใช้ค่าของช่วงสุดท้ายที่คะแนนดิบ >= ขอบล่างของช่วง"""
    if not points:
        return raw
    if raw < points[0][0]:
        return round(points[0][1] * (raw / points[0][0]) if points[0][0] > 0 else 0.0, 4)
    val = points[0][1]
    for lo, cal in points:
        if raw >= lo:
            val = cal
    return val


class Calibrator:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.table: dict = {}
        if self.path.exists():
            self.table = json.loads(self.path.read_text(encoding="utf-8"))

    def has(self, engine: str) -> bool:
        return engine in self.table

    def apply(self, fields: dict[str, FieldPrediction], engine: str) -> dict[str, FieldPrediction]:
        pts = self.table.get(engine, {}).get("points")
        if not pts:
            return fields
        for f in fields.values():
            f.raw_confidence = f.confidence
            f.confidence = apply_points(pts, f.confidence)
        return fields

    def save(self, engine: str, points: list[list[float]], meta: dict) -> None:
        self.table[engine] = {"points": points, **meta}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.table, ensure_ascii=False, indent=2), encoding="utf-8")
