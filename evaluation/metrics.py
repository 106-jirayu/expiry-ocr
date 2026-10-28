"""ตัวชี้วัดการประเมินผล

- CER (Character Error Rate)  = (S + D + I) / N    จาก Levenshtein distance ระดับตัวอักษร
- Field-Level Accuracy       = ฟิลด์ที่ค่า normalize ตรงกับ ground truth ทั้งหมด / จำนวนฟิลด์
- First-Pass Yield (FPY)     = ภาพที่ Auto-Approved (ไม่ต้องผ่านมือมนุษย์) / ภาพทั้งหมด
- Exception Rate             = ภาพที่ถูกส่งเข้า HITL Review Queue / ภาพทั้งหมด (= 1 - FPY)
- Auto-Approved Precision    = ภาพ Auto-Approved ที่ถูกทุกฟิลด์ / ภาพ Auto-Approved ทั้งหมด
                               (ตัวชี้วัดหลักของเป้า "ผลอัตโนมัติถูก ≥99%")
"""
from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(predictions: list[str | None], references: list[str]) -> float:
    """CER รวมทั้งชุด (micro-average) — prediction None = ลบทั้งหมด"""
    errors = sum(levenshtein(p or "", r) for p, r in zip(predictions, references))
    total = sum(len(r) for r in references)
    return errors / total if total else 0.0


def field_accuracy(pred: list[str | None], gt: list[str | None]) -> float:
    pairs = list(zip(pred, gt))
    return sum(p == g for p, g in pairs) / len(pairs) if pairs else 0.0


def ratio(n: int, d: int) -> float | None:
    return n / d if d else None
