"""Rule Validation Checklist — กฎธุรกิจตรวจความถูกต้องของวันที่

R01 required_exp       ต้องพบวันหมดอายุ
R02 date_format        รูปแบบวันที่ต้องอยู่ในรายการมาตรฐานที่อนุญาต (เช่น YYYY-MM-DD, DD/MM/YYYY)
R03 digit_range        เดือน 01-12, วัน 01-31, ปีอยู่ในช่วงที่เป็นไปได้
R04 calendar_valid     ต้องเป็นวันที่ที่มีอยู่จริง (เช่น 30/02 ไม่ผ่าน, 29/02 ต้องเป็นปีอธิกสุรทิน)
R05 exp_after_mfg      วันหมดอายุต้องไม่อยู่ก่อน (หรือเท่ากับ) วันผลิต
R06 mfg_not_future     วันผลิตต้องไม่อยู่ในอนาคต
R07 shelf_life         อายุสินค้า (EXP - MFG) ต้องไม่เกิน MAX_SHELF_LIFE_YEARS
R08 lot_format         Lot number ต้องเป็นตัวอักษร/ตัวเลข 3-20 ตัว
R08b                   (lot_format ใช้กับ lot ที่พบ label แต่อ่านค่าไม่ได้ด้วย)
R10 unclaimed_date     พบวันที่อื่นในภาพที่ไม่ได้จัดเป็น EXP/MFG (อาจหา label ไม่เจอ)
R09 not_expired        (warning) สินค้าหมดอายุแล้ว — แจ้งเตือนแต่ไม่บังคับส่ง HITL

severity: "error" = ไม่ผ่าน -> ส่ง HITL, "warning" = แจ้งเตือนเท่านั้น
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date

from app.extraction.kie import FieldPrediction

ALLOWED_FORMATS = {"YYYY-MM-DD", "DD/MM/YYYY", "DD/MM/YY", "DD MON YYYY", "MON YYYY", "MM/YYYY",
                   "DDMMYY", "DDMMYYYY"}
LOT_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-/]{2,19}$")


@dataclass
class RuleResult:
    rule_id: str
    name: str
    passed: bool
    field: str | None
    message: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return asdict(self)


def _d(f: FieldPrediction | None) -> date | None:
    return date.fromisoformat(f.value) if f and f.value else None


def validate(fields: dict[str, FieldPrediction], today: date | None = None,
             max_shelf_life_years: int = 10, min_year: int = 2000, max_year_ahead: int = 15,
             allowed_formats: set[str] = ALLOWED_FORMATS, extra_dates: list[str] | None = None) -> list[RuleResult]:
    today = today or date.today()
    out: list[RuleResult] = []
    exp, mfg, lot = fields.get("exp_date"), fields.get("mfg_date"), fields.get("lot_number")

    out.append(RuleResult("R01", "required_exp", exp is not None, "exp_date",
                          "พบวันหมดอายุ" if exp else "ไม่พบวันหมดอายุ (EXP) บนภาพ"))

    for f in (exp, mfg):
        if f is None:
            continue
        ok = f.date_format in allowed_formats
        out.append(RuleResult("R02", "date_format", ok, f.name,
                              f"รูปแบบ {f.date_format}" if ok else f"อ่าน/แยกรูปแบบวันที่ไม่ได้: {f.raw_text!r}"))
        if f.components is None:
            continue
        c = f.components
        problems = []
        if c.get("month") is None or not 1 <= c["month"] <= 12:
            problems.append(f"เดือน={c.get('month')} (ต้อง 01-12)")
        if c.get("day") is not None and not 1 <= c["day"] <= 31:
            problems.append(f"วัน={c['day']} (ต้อง 01-31)")
        if c.get("year") is None or not min_year <= c["year"] <= today.year + max_year_ahead:
            problems.append(f"ปี={c.get('year')} (ต้อง {min_year}-{today.year + max_year_ahead})")
        out.append(RuleResult("R03", "digit_range", not problems, f.name,
                              "ตัวเลขอยู่ในช่วงปกติ" if not problems else "; ".join(problems)))
        if not problems:
            ok = f.value is not None
            out.append(RuleResult("R04", "calendar_valid", ok, f.name,
                                  "วันที่มีอยู่จริง" if ok else f"ไม่มีวันที่นี้ในปฏิทิน: {f.raw_text!r}"))

    e, m = _d(exp), _d(mfg)
    if e and m:
        out.append(RuleResult("R05", "exp_after_mfg", e > m, "exp_date",
                              "EXP อยู่หลัง MFG" if e > m else f"EXP ({e}) ไม่อยู่หลัง MFG ({m})"))
        years = (e - m).days / 365.25
        ok = years <= max_shelf_life_years
        out.append(RuleResult("R07", "shelf_life", ok, "exp_date",
                              f"อายุสินค้า {years:.1f} ปี" if ok else
                              f"อายุสินค้า {years:.1f} ปี เกิน {max_shelf_life_years} ปี (อาจอ่านปีผิด)"))
    if m:
        ok = m <= today
        out.append(RuleResult("R06", "mfg_not_future", ok, "mfg_date",
                              "วันผลิตไม่อยู่ในอนาคต" if ok else f"วันผลิต {m} อยู่ในอนาคต"))
    if lot is not None:
        ok = bool(lot.value and LOT_RE.match(lot.value))
        out.append(RuleResult("R08", "lot_format", ok, "lot_number",
                              "รูปแบบ Lot ถูกต้อง" if ok else f"รูปแบบ Lot ผิดปกติ: {lot.value!r}"))
    if extra_dates:
        out.append(RuleResult("R10", "unclaimed_date", False, None,
                              "พบวันที่ที่ไม่รู้ว่าเป็น EXP หรือ MFG: " + ", ".join(repr(x) for x in extra_dates)))
    if e:
        ok = e >= today
        out.append(RuleResult("R09", "not_expired", ok, "exp_date",
                              "ยังไม่หมดอายุ" if ok else f"สินค้าหมดอายุแล้วเมื่อ {e}", severity="warning"))
    return out
