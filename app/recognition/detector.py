"""Text-line detector แบบ classical (OpenCV) สำหรับ engine ที่อ่านได้ทีละบรรทัด เช่น TrOCR

แนวทาง: morphological gradient -> Otsu -> closing แนวนอน -> contours -> กรองตามขนาด/สัดส่วน
(ถ้าต้องการความแม่นยำสูงขึ้นสามารถแทนด้วย DBNet / CRAFT ได้โดยคง interface เดิม)
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_text_lines(image: np.ndarray, pad: int = 6, min_h: int = 10) -> list[list[list[float]]]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kx = max(9, w // 40)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw_, bh = cv2.boundingRect(c)
        if bh < min_h or bw_ < bh * 1.2 or bh > h * 0.5:
            continue
        roi = bw[y:y + bh, x:x + bw_]
        if cv2.countNonZero(roi) / float(bw_ * bh) < 0.15:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw_ + pad), min(h, y + bh + pad)
        boxes.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    boxes.sort(key=lambda b: (b[0][1], b[0][0]))
    return [[[float(px), float(py)] for px, py in b] for b in boxes]


def crop_polygon(image: np.ndarray, polygon) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    h, w = image.shape[:2]
    return image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
