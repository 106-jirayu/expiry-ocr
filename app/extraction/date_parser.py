"""ตัวแยกวันที่จากข้อความ OCR — รองรับรูปแบบที่พบบ่อยบนบรรจุภัณฑ์

รูปแบบที่รองรับ (ชื่อ format ใช้ต่อในกฎธุรกิจ):
  YYYY-MM-DD   2025-12-31, 2025/12/31, 2025.12.31, 20251231 (ตัวเลขติดกัน), 202512.31 (ตัวคั่นหาย)
  DD/MM/YYYY   31/12/2025, 31-12-2025, 31.12.2025, 31 12 2025
  DD/MM/YY     31/12/25
  YY/MM/DD     25.12.31 (ฉลากเกาหลี/ญี่ปุ่น/จีน) — เลือกเมื่ออ่านแบบ DD/MM/YY แล้วไม่ใช่วันที่จริง
               หรือปีห่างจากวันนี้มากกว่าชัดเจน (ทำเครื่องหมาย ambiguous ให้คนยืนยัน)
  DD MON YYYY  31 DEC 2025, 31DEC25, 31 ธ.ค. 68
  MON DD YYYY  DEC 31 2025, DEC-31-2025, FEB/26/21, AUG312025 (แบบอเมริกัน)
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
# OCR มักอ่านจุดหาย: "ธ.ค", "ธค"
for _k, _v in list(TH_MONTHS.items()):
    if "." in _k:
        TH_MONTHS[_k.rstrip(".")] = _v
        TH_MONTHS[_k.replace(".", "")] = _v
_EN = "|".join(sorted(EN_MONTHS, key=len, reverse=True))
_TH = "|".join(re.escape(m) for m in sorted(TH_MONTHS, key=len, reverse=True))
MON = rf"(?P<mon>{_EN}|{_TH})"
SEP = r"[\-/.\s_]"

# ลำดับสำคัญ: เจาะจงมากก่อน
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("YYYY-MM-DD", re.compile(rf"(?<!\d)(?P<y>\d{{4}})(?P<s>{SEP})(?P<m>\d{{1,2}})(?P=s)(?P<d>\d{{1,2}})(?!\d)")),
    # ตัวเลขติดกันแบบปีขึ้นก่อน: 20251231 (จำกัดเดือน/วันให้อยู่ในช่วง เพื่อไม่ชนกับ DDMMYYYY)
    ("YYYY-MM-DD", re.compile(r"(?<!\d)(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])(?!\d)")),
    # OCR ทำตัวคั่นหายไป 1 ตัว: "202107.10", "2021.0710"
    ("YYYY-MM-DD", re.compile(r"(?<!\d)(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])[/.\-](?P<d>\d{2})(?!\d)")),
    ("YYYY-MM-DD", re.compile(r"(?<!\d)(?P<y>20\d{2})[/.\-](?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])(?!\d)")),
    ("DD MON YYYY", re.compile(rf"(?<!\d)(?P<d>\d{{1,2}}){SEP}?\s*{MON}\s*{SEP}?\s*(?P<y>\d{{4}}|\d{{2}})(?!\d)", re.I)),
    # แบบอเมริกัน เดือนขึ้นก่อน: "APR 14 2022", "DEC-28-2021", "FEB/26/21", "AUG292020"
    # (ต้องมีตัวคั่นระหว่างวันกับปี หรือเป็นปี 4 หลักติดกัน เพื่อไม่ชนกับ "DEC 2025")
    ("MON DD YYYY", re.compile(rf"(?<![A-Z]){MON}\s*{SEP}?\s*(?P<d>\d{{1,2}})\s*[\-/.,\s_]\s*(?P<y>\d{{4}}|\d{{2}})(?![\d/.\-])", re.I)),
    ("MON DD YYYY", re.compile(rf"(?<![A-Z]){MON}\s*(?P<d>\d{{2}})(?P<y>(?:19|20)\d{{2}})(?!\d)", re.I)),
    # ตัวคั่นทั้งสองตำแหน่งต้องเป็นตัวเดียวกัน (กันกรณี "EXP 1 16/07/28" ถูกอ่านเป็น 1/16/07)
    ("DD/MM/YYYY", re.compile(rf"(?<!\d)(?P<d>\d{{1,2}})(?P<s>{SEP})(?P<m>\d{{1,2}})(?P=s)(?P<y>\d{{4}})(?!\d)")),
    # OCR ทำตัวคั่นหายไป 1 ตัว: "11.112025", "1111.2025", "0710 2569"
    ("DD/MM/YYYY", re.compile(r"(?<!\d)(?P<d>\d{1,2})[/.\-](?P<m>\d{2})(?P<y>(?:20|25)\d{2})(?!\d)")),
    ("DD/MM/YYYY", re.compile(r"(?<!\d)(?P<d>\d{2})(?P<m>\d{2})[/.\-\s](?P<y>(?:20|25)\d{2})(?!\d)")),
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
    ambiguous: bool = False       # ตีความได้มากกว่า 1 แบบ (เช่น 21.06.25) — ควรให้คนยืนยัน
    alternative: str | None = None  # ค่าอีกแบบ (ISO) เมื่อ ambiguous

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


def _maybe_year_first(pd: ParsedDate, g: dict, today: date | None) -> ParsedDate:
    """"21.06.25" อาจเป็น 21 มิ.ย. 2025 (DD/MM/YY) หรือ 2021-06-25 (YY/MM/DD แบบเอเชียตะวันออก)

    ใช้แบบ YY/MM/DD เมื่อแบบ DD/MM/YY ไม่ใช่วันที่จริง หรือปีของแบบ YY/MM/DD ใกล้วันนี้กว่าชัดเจน (> 2 ปี)
    """
    today = today or date.today()
    alt = ParsedDate("YY/MM/DD", pd.start, pd.end, pd.raw, int(g["y"]), pd.month, 2000 + int(g["d"]))
    a, p = alt.to_date(), pd.to_date()
    if a is None:
        return pd
    if p is None or abs(p.year - today.year) - abs(a.year - today.year) > 2:
        if p is not None:
            alt.ambiguous, alt.alternative = True, p.isoformat()
        return alt
    return pd


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
            pd = ParsedDate(fmt, s, e, m.group(0), day, month, year, day_implied=day is None, buddhist_era=be)
            if fmt == "DD/MM/YY":
                pd = _maybe_year_first(pd, g, today)
            found.append(pd)
            taken.append((s, e))
    return sorted(found, key=lambda p: p.start)
