"""วาดกรอบไฮไลท์ตำแหน่งที่ระบบอ่านได้บนภาพต้นฉบับ (ใช้ทั้งใน UI และ API)"""
from __future__ import annotations

import cv2
import numpy as np

LABELS = {"exp_date": "EXP", "mfg_date": "MFG", "lot_number": "LOT"}


def draw_annotations(image: np.ndarray, fields: list[dict], threshold: float = 0.90,
                     all_lines: list[dict] | None = None) -> np.ndarray:
    """fields: [{"name","value","confidence","polygon"}] — สีเขียว = มั่นใจ, ส้ม/แดง = ต่ำกว่าเกณฑ์"""
    out = image.copy()
    scale = max(1.0, max(out.shape[:2]) / 900)
    thick = max(2, int(2 * scale))
    for ln in all_lines or []:
        if ln.get("polygon"):
            pts = np.asarray(ln["polygon"], dtype=np.int32)
            cv2.polylines(out, [pts], True, (200, 200, 200), 1)
    for f in fields:
        if not f.get("polygon"):
            continue
        conf = f.get("confidence") or 0.0
        color = (60, 180, 60) if conf >= threshold else ((0, 165, 255) if conf >= 0.7 else (40, 40, 220))
        pts = np.asarray(f["polygon"], dtype=np.int32)
        cv2.polylines(out, [pts], True, color, thick)
        label = f"{LABELS.get(f['name'], f['name'])} {f.get('value') or '?'} ({conf:.0%})"
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        fs = 0.5 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y_top = max(0, y - th - 8)
        cv2.rectangle(out, (x, y_top), (x + tw + 6, y_top + th + 8), color, -1)
        cv2.putText(out, label, (x + 3, y_top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()
