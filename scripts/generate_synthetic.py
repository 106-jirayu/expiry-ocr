"""สร้างชุดข้อมูลสังเคราะห์ (ภาพฉลากวันหมดอายุ + ground truth) สำหรับสาธิตและประเมินผล

ภาพแต่ละภาพ:
  - สุ่มรูปแบบวันที่/label (EXP, BB, BEST BEFORE, USE BY, MFG, MFD, LOT, BATCH ...)
  - สุ่มพื้นหลัง, สีหมึก, ฟอนต์, ข้อความรบกวน (NET WT, ส่วนผสม ...)
  - ผ่าน Data Augmentation (rotation ±10°, blur, erosion, dilation, low contrast, noise)
  - ฝัง ground truth (ข้อความ + polygon) ไว้ใน PNG metadata key "ocr_gt" (ใช้โดย MockOCREngine)
  - เขียน manifest.jsonl สำหรับ evaluation/evaluate.py

ใช้งาน:  python -m scripts.generate_synthetic --out data/synthetic --n 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from app.intake.augmentation import AugmentationPolicy, augment
from app.intake.preprocessing import transform_points

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
EXP_LABELS = ["EXP", "EXP:", "EXP.", "BB", "BBE", "BEST BEFORE", "USE BY", "EXPIRY DATE:"]
MFG_LABELS = ["MFG", "MFG:", "MFD", "MFD:", "PROD:", "MFG DATE:"]
LOT_LABELS = ["LOT", "LOT:", "LOT NO.", "BATCH:", "L:"]
DISTRACTORS = ["NET WT 500 g", "PRODUCT OF THAILAND", "KEEP REFRIGERATED", "INGREDIENTS: SUGAR, MILK",
               "STORE IN A COOL DRY PLACE", "BARCODE 8850000123", "NET 250 ML", "HALAL"]


def load_fonts(size: int) -> list:
    fonts = []
    for p in FONT_CANDIDATES:
        try:
            fonts.append(ImageFont.truetype(p, size))
        except OSError:
            continue
    return fonts or [ImageFont.load_default(size=size)]


def fmt_date(d: date, style: str) -> str:
    return {
        "DD/MM/YYYY": d.strftime("%d/%m/%Y"),
        "DD-MM-YYYY": d.strftime("%d-%m-%Y"),
        "DD.MM.YYYY": d.strftime("%d.%m.%Y"),
        "DD/MM/YY": d.strftime("%d/%m/%y"),
        "YYYY-MM-DD": d.isoformat(),
        "DD MON YYYY": f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}",
        "DDMONYY": f"{d.day:02d}{MONTHS[d.month - 1]}{d.year % 100:02d}",
        "BE DD/MM/YYYY": f"{d.day:02d}/{d.month:02d}/{d.year + 543}",
    }[style]


STYLES = ["DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY", "DD/MM/YY", "YYYY-MM-DD", "DD MON YYYY", "DDMONYY",
          "BE DD/MM/YYYY"]


def random_sample(rng: np.random.Generator, today: date):
    mfg = today - timedelta(days=int(rng.integers(1, 400)))
    exp = mfg + timedelta(days=int(rng.integers(30, 365 * 3)))
    style = STYLES[int(rng.integers(len(STYLES)))]
    lot = "".join(rng.choice(list("ABCDEFGHJKMNPRTUVWXY"), 1)) + "".join(rng.choice(list("0123456789"), int(rng.integers(3, 6)))) \
        + ("".join(rng.choice(list("ABCDEFGHJK"), 1)) if rng.random() < 0.5 else "")
    has_mfg = rng.random() < 0.8
    has_lot = rng.random() < 0.8
    lines = []
    exp_text = fmt_date(exp, style)
    mfg_text = fmt_date(mfg, style)
    if has_mfg:
        lines.append(("mfg_date", f"{MFG_LABELS[int(rng.integers(len(MFG_LABELS)))]} {mfg_text}", mfg_text))
    lines.append(("exp_date", f"{EXP_LABELS[int(rng.integers(len(EXP_LABELS)))]} {exp_text}", exp_text))
    if has_lot:
        lines.append(("lot_number", f"{LOT_LABELS[int(rng.integers(len(LOT_LABELS)))]} {lot}", lot))
    if rng.random() < 0.3 and len(lines) >= 2:  # บางฉลากพิมพ์ EXP กับ LOT บรรทัดเดียวกัน
        a, b = lines[-2], lines[-1]
        lines = lines[:-2] + [("combo", f"{a[1]}  {b[1]}", None)]
    distract = [d for d in DISTRACTORS if rng.random() < 0.25]
    gt = {"exp_date": exp.isoformat(), "mfg_date": mfg.isoformat() if has_mfg else None,
          "lot_number": lot if has_lot else None, "exp_text": exp_text,
          "mfg_text": mfg_text if has_mfg else None, "lot_text": lot if has_lot else None, "style": style}
    return gt, [ln[1] for ln in lines], distract


def render(rng: np.random.Generator, text_lines: list[str], distract: list[str]):
    size = int(rng.integers(26, 46))
    fonts = load_fonts(size)
    font = fonts[int(rng.integers(len(fonts)))]
    bg = [(250, 250, 248), (235, 220, 190), (210, 212, 215), (255, 255, 255), (245, 235, 225), (40, 40, 45)]
    bgc = bg[int(rng.integers(len(bg)))]
    dark_bg = sum(bgc) < 300
    ink = (235, 235, 235) if dark_bg else [(20, 20, 20), (30, 30, 120), (90, 20, 20)][int(rng.integers(3))]
    all_lines = distract[: len(distract) // 2] + text_lines + distract[len(distract) // 2:]
    W = int(max(font.getbbox(t)[2] for t in all_lines) + rng.integers(60, 200))
    H = int(len(all_lines) * size * 1.6 + rng.integers(60, 160))
    img = Image.new("RGB", (W, H), bgc)
    draw = ImageDraw.Draw(img)
    y = int(rng.integers(25, 60))
    x0 = int(rng.integers(20, 50))
    out_lines = []
    for t in all_lines:
        draw.text((x0, y), t, font=font, fill=ink)
        l, tp, r, b = draw.textbbox((x0, y), t, font=font)
        out_lines.append({"text": t, "polygon": [[l - 4, tp - 4], [r + 4, tp - 4], [r + 4, b + 4], [l - 4, b + 4]]})
        y += int(size * rng.uniform(1.35, 1.8))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), out_lines


def save_with_gt(path: Path, img_bgr: np.ndarray, lines: list[dict]):
    info = PngImagePlugin.PngInfo()
    info.add_text("ocr_gt", json.dumps({"lines": lines}, ensure_ascii=False))
    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(path, pnginfo=info)


def generate(out: Path, n: int, seed: int, today: date | None = None, policy: AugmentationPolicy | None = None):
    rng = np.random.default_rng(seed)
    today = today or date.today()
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        gt, text_lines, distract = random_sample(rng, today)
        img, lines = render(rng, text_lines, distract)
        img, m, applied = augment(img, rng, policy)
        for ln in lines:
            ln["polygon"] = transform_points(ln["polygon"], m).round(1).tolist()
        name = f"img_{i:04d}.png"
        save_with_gt(out / "images" / name, img, lines)
        rows.append({"image": f"images/{name}", "ground_truth": gt, "augmentations": applied,
                     "lines": [ln["text"] for ln in lines]})
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (ค่าเริ่มต้น = วันนี้)")
    a = ap.parse_args()
    rows = generate(Path(a.out), a.n, a.seed, date.fromisoformat(a.today) if a.today else None)
    print(f"สร้าง {len(rows)} ภาพที่ {a.out}/images และ {a.out}/manifest.jsonl")


if __name__ == "__main__":
    main()
