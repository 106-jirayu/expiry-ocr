"""Key Information Extraction (KIE) — สกัด EXP Date, MFG Date, Lot Number เป็น Structured JSON

ขั้นตอน
 1. Normalize ข้อความ (ตัวพิมพ์ใหญ่) โดยคงความยาวเดิม -> index ตรงกับ char_confidences
 2. หา keyword label (EXP/BB/USE BY/หมดอายุ, MFG/MFD/ผลิต, LOT/BATCH/L:) รองรับ OCR สะกดผิดที่พบบ่อย
 3. ในช่วงข้อความหลัง label แก้ตัวอักษรที่ OCR สับสนในบริบทตัวเลข (O->0, I->1, S->5, B->8 ...)
    แล้วหาวันที่ด้วย date_parser
 4. ถ้าไม่พบ label แต่มีวันที่ลอย ๆ: วันที่ล่าสุด = EXP, วันที่เก่าสุด = MFG (ทำเครื่องหมาย inferred)
 5. ความมั่นใจระดับฟิลด์ = min(char confidence ในช่วงค่า) x penalty (ดู rules/confidence.py)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from app.extraction.date_parser import ParsedDate, find_dates
from app.recognition.base import TextLine

FIELDS = ("exp_date", "mfg_date", "lot_number")

LABELS = {
    "exp_date": re.compile(
        r"(?<![A-Z])([B8][E3][S5]T\s*[B8][E3]F[O0]RE(\s*END)?|U[S5]E\s*[B8]Y|EXP(IRY|IRES|IRE)?(\s*DATE)?|EXD|(?<![A-Z0-9])[B8]{2}E?(?![A-Z])|"
        r"วันหมดอายุ|หมดอายุ|ควรบริโภคก่อน|บริโภคก่อน|ควรใช้ก่อน)\s*[:.\-]?",
        re.I),
    "mfg_date": re.compile(
        r"(?<![A-Z])(MFG|ME?F[G6D0O]|MANUFACTURED(\s*DATE)?|MFD|PROD(UCTION)?(\s*DATE)?(?![A-Z])|PKD|"
        r"วันผลิต|ผลิต|ผลิตเมื่อ)\s*[:.\-]?",
        re.I),
    "lot_number": re.compile(
        r"(?<![A-Z])(L[O0]T\s*(N[O0]\.?)?|[B8]ATCH(\s*N[O0]\.?)?|ล็อต|ล๊อต|รุ่นการผลิต|L(?=\s*[:.#]))\s*[:.#]?\s*",
        re.I),
}
LOT_VALUE = re.compile(r"[A-Z0-9][A-Z0-9\-/]{2,19}")

# ตัวอักษรที่ OCR มักอ่านสลับกับตัวเลข — แก้เฉพาะเมื่ออยู่ติดตัวเลข/ตัวคั่นวันที่
DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "!": "1",
             "S": "5", "B": "8", "Z": "2", "G": "6", "g": "9", "T": "7",
             ",": "."}  # จุลภาคระหว่างตัวเลข = ตัวคั่นวันที่ที่ OCR อ่านเพี้ยน

# ชื่อเดือนที่ OCR อ่านตัวอักษรเป็นตัวเลข
MONTH_FIX = {"0CT": "OCT", "N0V": "NOV", "5EP": "SEP", "AU6": "AUG", "8EC": "DEC", "JU1": "JUL", "MA8": "MAR"}


@dataclass
class FieldPrediction:
    name: str
    raw_text: str | None = None               # ข้อความดิบที่อ่านได้ (ช่วงค่า)
    value: str | None = None                  # ค่า normalize: ISO date หรือ lot code
    confidence: float = 0.0
    char_confidences: list[float] = field(default_factory=list)
    date_format: str | None = None
    components: dict | None = None            # {"day","month","year"} สำหรับกฎตรวจช่วงตัวเลข
    polygon: list | None = None
    line_index: int | None = None
    line_text: str | None = None
    label: str | None = None
    ocr_fixes: int = 0
    inferred: bool = False                    # ไม่มี label กำกับ เดาจากตำแหน่ง/ลำดับ
    agreement: float = 1.0                    # สัดส่วน pass ที่เห็นตรงกัน (ensemble)
    raw_confidence: float | None = None       # คะแนนก่อน calibration (ถ้ามี)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        d["char_confidences"] = [round(c, 4) for c in self.char_confidences]
        return d


def fix_digits(segment: str) -> tuple[str, int]:
    """แก้ตัวอักษรที่น่าจะเป็นตัวเลขในส่วนที่ดูเป็นวันที่ (ไม่แตะชื่อเดือน)

    ทำซ้ำจนไม่มีการเปลี่ยนแปลง เพื่อรองรับกรณีสับสนติดกันหลายตัว เช่น "1O/OI/2027"
    """
    for wrong, right in MONTH_FIX.items():
        segment = segment.replace(wrong, right)
    chars = list(segment)
    month_spans = [m.span() for m in re.finditer(
        r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEPT?|OCT|NOV|DEC", segment, re.I)]

    def in_month(i):
        return any(s <= i < e for s, e in month_spans)

    def near_digit(i):
        # ติดกับตัวเลขโดยตรง หรือคั่นด้วยตัวคั่นวันที่ 1 ตัวแล้วเป็นตัวเลข
        for step in (-1, 1):
            j = i + step
            if 0 <= j < len(chars):
                if chars[j].isdigit():
                    return True
                k = j + step
                if chars[j] in "/-." and 0 <= k < len(chars) and chars[k].isdigit():
                    return True
        return False

    fixes, changed = 0, True
    while changed:
        changed = False
        for i, ch in enumerate(chars):
            if ch in DIGIT_FIX and not in_month(i) and near_digit(i):
                chars[i] = DIGIT_FIX[ch]
                fixes += 1
                changed = True
    return "".join(chars), fixes


def _span_conf(line: TextLine, start: int, end: int) -> list[float]:
    return line.char_confidences[start:end] or [line.confidence]


def _date_prediction(name: str, line: TextLine, idx: int, offset: int, pd: ParsedDate,
                     fixes: int, label: str | None, inferred: bool) -> FieldPrediction:
    s, e = offset + pd.start, offset + pd.end
    confs = _span_conf(line, s, e)
    d = pd.to_date(prefer_end_of_month=(name == "exp_date"))
    return FieldPrediction(
        name=name, raw_text=line.text[s:e], value=d.isoformat() if d else None,
        confidence=min(confs), char_confidences=confs, date_format=pd.format,
        components={"day": pd.day, "month": pd.month, "year": pd.year,
                    "day_implied": pd.day_implied, "buddhist_era": pd.buddhist_era},
        polygon=line.polygon, line_index=idx, line_text=line.text, label=label,
        ocr_fixes=fixes, inferred=inferred,
    )


def _unreadable(name: str, line: TextLine, idx: int, start: int, end: int, label: str) -> FieldPrediction:
    """พบ label (เช่น "EXP") แต่แยกค่าไม่ได้ -> ส่งเป็นฟิลด์ความมั่นใจ 0 เพื่อให้เข้า HITL เสมอ"""
    return FieldPrediction(name=name, raw_text=line.text[start:end].strip() or None, value=None,
                           confidence=0.0, char_confidences=[0.0], polygon=line.polygon,
                           line_index=idx, line_text=line.text, label=label)


def _label_hits(upper: str) -> list[tuple[int, int, str, str]]:
    hits = []
    for fname, pat in LABELS.items():
        for m in pat.finditer(upper):
            hits.append((m.start(), m.end(), fname, m.group(0).strip()))
    hits.sort()
    # ตัดตัวที่ซ้อนกัน (เช่น "BB" ภายใน "BEST BEFORE")
    out = []
    for h in hits:
        if out and h[0] < out[-1][1]:
            if (h[1] - h[0]) > (out[-1][1] - out[-1][0]):
                out[-1] = h
            continue
        out.append(h)
    return out


def extract_fields(lines: list[TextLine], today: date | None = None) -> dict[str, FieldPrediction]:
    found: dict[str, FieldPrediction] = {}
    floating: list[tuple[int, TextLine, ParsedDate, int, int]] = []

    for idx, line in enumerate(lines):
        upper = line.text.upper()
        hits = _label_hits(upper)
        bounds = [h[0] for h in hits[1:]] + [len(upper)]
        covered_until = 0
        for (s, e, fname, label), seg_end in zip(hits, bounds):
            if fname in found and found[fname].value is not None:
                continue   # ใช้ค่าแรกที่อ่านได้ (label ที่อ่านค่าไม่ได้ ยอมให้ label ถัดไปแทนที่)
            seg = upper[e:seg_end]
            if fname == "lot_number":
                m = LOT_VALUE.search(seg)
                if m and not find_dates(seg[m.start():m.end()], today):
                    vs, ve = e + m.start(), e + m.end()
                    confs = _span_conf(line, vs, ve)
                    found[fname] = FieldPrediction(
                        name=fname, raw_text=line.text[vs:ve], value=m.group(0),
                        confidence=min(confs), char_confidences=confs, polygon=line.polygon,
                        line_index=idx, line_text=line.text, label=label)
                    covered_until = max(covered_until, ve)
                else:
                    found[fname] = _unreadable(fname, line, idx, e, seg_end, label)
                continue
            fixed, fixes = fix_digits(seg)
            dates = find_dates(fixed, today)
            if dates:
                pd = dates[0]
                seg_fixes = sum(1 for a, b in zip(seg[pd.start:pd.end], fixed[pd.start:pd.end]) if a != b)
                pd.raw = line.text[e + pd.start:e + pd.end]
                found[fname] = _date_prediction(fname, line, idx, e, pd, seg_fixes, label, False)
                covered_until = max(covered_until, e + pd.end)
            else:
                found[fname] = _unreadable(fname, line, idx, e, seg_end, label)
        # วันที่ที่ไม่มี label กำกับ
        if not hits:
            fixed, _ = fix_digits(upper)
            for pd in find_dates(fixed, today):
                seg_fixes = sum(1 for a, b in zip(upper[pd.start:pd.end], fixed[pd.start:pd.end]) if a != b)
                floating.append((idx, line, pd, 0, seg_fixes))

    # เติมฟิลด์ที่ขาดจากวันที่ลอย ๆ
    if floating and ("exp_date" not in found or "mfg_date" not in found):
        def key(item):
            d = item[2].to_date()
            return d or date.min
        floating.sort(key=key)
        if "exp_date" not in found:
            idx, line, pd, off, fx = floating.pop(-1)
            found["exp_date"] = _date_prediction("exp_date", line, idx, off, pd, fx, None, True)
        if "mfg_date" not in found and floating:
            idx, line, pd, off, fx = floating.pop(0)
            found["mfg_date"] = _date_prediction("mfg_date", line, idx, off, pd, fx, None, True)
    return found


def to_structured_json(fields: dict[str, FieldPrediction]) -> dict:
    """รูปแบบ JSON ที่ส่งต่อ downstream"""
    return {name: (fields[name].to_dict() if name in fields else None) for name in FIELDS}
