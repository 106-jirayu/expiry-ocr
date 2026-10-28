"""ตัวแยกวันที่จากข้อความ OCR — รองรับรูปแบบที่พบบ่อยบนบรรจุภัณฑ์

รูปแบบที่รองรับ (ชื่อ format ใช้ต่อในกฎธุรกิจ):
  YYYY-MM-DD   2025-12-31, 2025/12/31, 2025.12.31
  DD/MM/YYYY   31/12/2025, 31-12-2025, 31.12.2025, 31 12 2025
  DD/MM/YY     31/12/25
  DD MON YYYY  31 DEC 2025, 31DEC25, 31 ธ.ค. 68
  MON YYYY     DEC 2025 (วันที่ = วันสุดท้ายของเดือนสำหรับ EXP, วันที่ 1 สำหรับ MFG)
  MM/YYYY      12/2025
  DDMMYY       311225, 31122025 (ตัวเลขติดกัน)

ปี พ.ศ. (เช่น 2568 หรือ 68) แปลงเป็น ค.ศ. อัตโนมัติ
parser จะ "ไม่" ตรวจว่าวันที่มีจริงหรือไม่ — แค่แยกส่วนประกอบ ส่วนการตรวจเป็นหน้าที่ของ rules engine
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

EN_MONTHS = {m.upper(): i for i, m in enumerate(calendar.month_abbr) if m}
EN_MONTHS.update({m.upper(): i for i, m in enumerate(calendar.month_name) if m})
EN_MONTHS["SEPT"] = 9
TH_MONTHS = {
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
}
_EN = "|".join(sorted(EN_MONTHS, key=len, reverse=True))
_TH = "|".join(re.escape(m) for m in sorted(TH_MONTHS, key=len, reverse=True))
MON = rf"(?P<mon>{_EN}|{_TH})"
SEP = r"[\-/.\s]"

# ลำดับสำคัญ: เจาะจงมากก่อน
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("YYYY-MM-DD", re.compile(rf"(?<!\d)(?P<y>\d{{4}})(?P<s>{SEP})(?P<m>\d{{1,2}})(?P=s)(?P<d>\d{{1,2}})(?!\d)")),
    ("DD MON YYYY", re.compile(rf"(?<!\d)(?P<d>\d{{1,2}}){SEP}?\s*{MON}\s*{SEP}?\s*(?P<y>\d{{4}}|\d{{2}})(?!\d)", re.I)),
    # ตัวคั่นทั้งสองตำแหน่งต้องเป็นตัวเดียวกัน (กันกรณี "EXP 1 16/07/28" ถูกอ่านเป็น 1/16/07)
    ("DD/MM/YYYY", re.compile(rf"(?<!\d)(?P<d>\d{{1,2}})(?P<s>{SEP})(?P<m>\d{{1,2}})(?P=s)(?P<y>\d{{4}})(?!\d)")),
    ("DD/MM/YY", re.compile(rf"(?<!\d)(?P<d>\d{{1,2}})(?P<s>{SEP})(?P<m>\d{{1,2}})(?P=s)(?P<y>\d{{2}})(?![\d/.\-])")),
    ("MON YYYY", re.compile(rf"(?<![A-Z]){MON}\s*{SEP}?\s*(?P<y>\d{{4}}|\d{{2}})(?!\d)", re.I)),
    ("MM/YYYY", re.compile(rf"(?<!\d)(?P<m>\d{{1,2}})[/.\-](?P<y>\d{{4}})(?!\d)")),
    ("DDMMYYYY", re.compile(r"(?<!\d)(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{4})(?!\d)")),
    ("DDMMYY", re.compile(r"(?<!\d)(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{2})(?!\d)")),
]


@dataclass
class ParsedDate:
    format: str
    start: int
    end: int
    raw: str
    day: int | None
    month: int | None
    year: int | None
    day_implied: bool = False
    buddhist_era: bool = False

    def to_date(self, prefer_end_of_month: bool = True) -> date | None:
        """คืน date ถ้าเป็นวันที่ที่มีอยู่จริง ไม่เช่นนั้นคืน None"""
        if not (self.year and self.month):
            return None
        day = self.day
        if day is None:
            if not 1 <= self.month <= 12:
                return None
            day = calendar.monthrange(self.year, self.month)[1] if prefer_end_of_month else 1
        try:
            return date(self.year, self.month, day)
        except ValueError:
            return None


def normalize_year(y: str, today: date | None = None) -> tuple[int, bool]:
    """แปลงปีเป็น ค.ศ. 4 หลัก คืน (ปี, เป็น พ.ศ. หรือไม่)"""
    today = today or date.today()
    v = int(y)
    if len(y) == 4:
        return (v - 543, True) if v >= 2400 else (v, False)
    ce = 2000 + v
    if abs(ce - today.year) <= 15:
        return ce, False
    be = 2500 + v - 543
    if abs(be - today.year) <= 15:
        return be, True
    return ce, False


def _month(m) -> int | None:
    if m is None:
        return None
    key = m.upper()
    return EN_MONTHS.get(key) or TH_MONTHS.get(m)


def find_dates(text: str, today: date | None = None) -> list[ParsedDate]:
    """หาวันที่ทุกตัวในข้อความ (ไม่ซ้อนทับกัน) เรียงตามตำแหน่ง"""
    found: list[ParsedDate] = []
    taken: list[tuple[int, int]] = []
    for fmt, pat in PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            g = m.groupdict()
            year, be = normalize_year(g["y"], today)
            month = int(g["m"]) if g.get("m") else _month(g.get("mon"))
            day = int(g["d"]) if g.get("d") else None
            found.append(ParsedDate(fmt, s, e, m.group(0), day, month, year,
                                    day_implied=day is None, buddhist_era=be))
            taken.append((s, e))
    return sorted(found, key=lambda p: p.start)
