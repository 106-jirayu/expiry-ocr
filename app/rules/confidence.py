"""Field-Level Confidence Calculation

confidence(field) = min(char_conf ในช่วงค่า)              <- ตัวอักษรที่อ่อนที่สุดเป็นตัวกำหนด
                    x FIX_PENALTY ** ocr_fixes             <- ต้องแก้ O->0 ฯลฯ แปลว่า OCR ไม่ชัด
                    x INFERRED_PENALTY (ถ้าไม่มี label)     <- เดาว่าเป็น EXP/MFG จากลำดับวันที่
                    x agreement (ensemble)                 <- สัดส่วนรอบที่อ่านได้ค่าตรงกัน

เหตุผลที่ใช้ min แทน mean: วันที่ผิด 1 ตัวอักษร = ผิดทั้งฟิลด์ ดังนั้นความมั่นใจของฟิลด์
ไม่ควรสูงกว่าความมั่นใจของตัวอักษรที่แย่ที่สุด (ช่วยให้ Auto-Approved แม่นยำระดับ ≥99%)
"""
from __future__ import annotations

from app.extraction.kie import FieldPrediction

FIX_PENALTY = 0.93
INFERRED_PENALTY = 0.88
REQUIRED_FIELDS = ("exp_date",)


def field_confidence(f: FieldPrediction) -> float:
    if not f.char_confidences:
        return 0.0
    conf = min(f.char_confidences)
    conf *= FIX_PENALTY ** f.ocr_fixes
    if f.inferred:
        conf *= INFERRED_PENALTY
    conf *= f.agreement
    return round(max(0.0, min(1.0, conf)), 4)


def apply_confidence(fields: dict[str, FieldPrediction]) -> dict[str, FieldPrediction]:
    for f in fields.values():
        f.confidence = field_confidence(f)
    return fields


def document_confidence(fields: dict[str, FieldPrediction], required=REQUIRED_FIELDS) -> float:
    """ความมั่นใจของทั้งภาพ = ต่ำสุดของฟิลด์ที่มีอยู่ (ฟิลด์บังคับที่หายไป = 0)"""
    confs = [fields[n].confidence if n in fields else 0.0 for n in required]
    confs += [f.confidence for n, f in fields.items() if n not in required]
    return round(min(confs), 4) if confs else 0.0
