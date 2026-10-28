"""Multi-pass Ensemble / Consensus — กลไกหลักที่ดันความแม่นยำของ Auto-Approved ให้ถึง ≥99%

อ่านภาพหลายรอบ (หลาย preprocessing variant และ/หรือ หลาย engine) แล้วลงคะแนนเสียงระดับฟิลด์:
  - ค่าที่ได้เสียงมากสุดเป็นผลลัพธ์ (เสมอกัน -> เลือกค่าที่ความมั่นใจรวมสูงกว่า)
  - agreement = จำนวนรอบที่เห็นด้วย / จำนวนรอบทั้งหมด
  - ถ้า agreement < 1 ระบบจะไม่ auto-approve (ดู router) => ความผิดพลาดแบบ "มั่นใจแต่ผิด"
    ของรอบใดรอบหนึ่งจะถูกจับได้เมื่ออีกรอบอ่านไม่ตรงกัน
"""
from __future__ import annotations

from collections import defaultdict

from app.extraction.kie import FIELDS, FieldPrediction


def merge_passes(passes: list[dict[str, FieldPrediction]]) -> dict[str, FieldPrediction]:
    if not passes:
        return {}
    if len(passes) == 1:
        return passes[0]
    n = len(passes)
    merged: dict[str, FieldPrediction] = {}
    for name in FIELDS:
        votes: dict[str | None, list[FieldPrediction]] = defaultdict(list)
        missing = 0
        for p in passes:
            f = p.get(name)
            if f is None:
                missing += 1
            else:
                votes[f.value if f.value is not None else f"RAW:{f.raw_text}"].append(f)
        if not votes:
            continue
        best_key = max(votes, key=lambda k: (len(votes[k]), sum(min(f.char_confidences or [0]) for f in votes[k])))
        group = votes[best_key]
        rep = max(group, key=lambda f: min(f.char_confidences or [0]))
        # ใช้ char confidence เฉลี่ยของรอบที่เห็นตรงกัน (ตัวแทนความชัดของภาพ)
        if all(len(f.char_confidences) == len(rep.char_confidences) for f in group):
            rep.char_confidences = [sum(c) / len(c) for c in zip(*[f.char_confidences for f in group])]
        rep.agreement = len(group) / n
        merged[name] = rep
    return merged
