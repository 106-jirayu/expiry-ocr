"""Data Augmentation สำหรับสร้างข้อมูลฝึก / ทดสอบความทนทานของโมเดล

- Random Rotation  (-10 ถึง 10 องศา)
- Gaussian Blur
- Erosion  (ตัวอักษรบางลง เหมือนหมึกจาง)
- Dilation (ตัวอักษรหนาขึ้น เหมือนหมึกเยิ้ม / dot-matrix)
- ปัจจัยเพิ่มเติม: low contrast, gaussian noise

ทุกฟังก์ชันรับ/คืน np.ndarray (BGR หรือ grayscale) และรับ rng เพื่อให้ผลซ้ำได้ (reproducible)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from app.intake.preprocessing import rotate


def random_rotation(img: np.ndarray, rng: np.random.Generator, min_deg: float = -10, max_deg: float = 10):
    """คืน (ภาพ, affine matrix, มุม) — matrix ใช้แปลงพิกัด bbox ของ ground truth"""
    angle = float(rng.uniform(min_deg, max_deg))
    out, m = rotate(img, angle)
    return out, m, angle


def gaussian_blur(img: np.ndarray, rng: np.random.Generator, max_sigma: float = 2.0) -> np.ndarray:
    sigma = float(rng.uniform(0.3, max_sigma))
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _morph_kernel(rng: np.random.Generator):
    k = int(rng.integers(2, 4))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _text_is_dark(img: np.ndarray) -> bool:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) > 127


def erosion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ทำให้ 'ตัวอักษร' บางลง (คำนึงว่าตัวอักษรเข้มบนพื้นสว่าง -> ต้อง dilate พื้น)"""
    k = _morph_kernel(rng)
    return cv2.dilate(img, k) if _text_is_dark(img) else cv2.erode(img, k)


def dilation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ทำให้ 'ตัวอักษร' หนาขึ้น"""
    k = _morph_kernel(rng)
    return cv2.erode(img, k) if _text_is_dark(img) else cv2.dilate(img, k)


def low_contrast(img: np.ndarray, rng: np.random.Generator, min_alpha: float = 0.35) -> np.ndarray:
    alpha = float(rng.uniform(min_alpha, 0.8))
    mean = float(np.mean(img))
    return np.clip((img.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, rng: np.random.Generator, max_std: float = 12.0) -> np.ndarray:
    noise = rng.normal(0, rng.uniform(2, max_std), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@dataclass
class AugmentationPolicy:
    p_rotation: float = 0.7
    p_blur: float = 0.4
    p_erosion: float = 0.15
    p_dilation: float = 0.15
    p_low_contrast: float = 0.3
    p_noise: float = 0.3
    max_blur_sigma: float = 2.0


def augment(img: np.ndarray, rng: np.random.Generator, policy: AugmentationPolicy | None = None):
    """สุ่ม augmentation ตาม policy

    คืน (ภาพ, affine matrix สำหรับแปลงพิกัด, รายการ augmentation ที่ใช้)
    """
    p = policy or AugmentationPolicy()
    matrix = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    applied: list[str] = []
    if rng.random() < p.p_rotation:
        img, matrix, ang = random_rotation(img, rng)
        applied.append(f"rotation({ang:+.1f})")
    ops: list[tuple[float, str, Callable]] = [
        (p.p_erosion, "erosion", erosion),
        (p.p_dilation, "dilation", dilation),
        (p.p_blur, "gaussian_blur", lambda i, r: gaussian_blur(i, r, p.max_blur_sigma)),
        (p.p_low_contrast, "low_contrast", low_contrast),
        (p.p_noise, "gaussian_noise", gaussian_noise),
    ]
    for prob, name, fn in ops:
        if rng.random() < prob:
            img = fn(img, rng)
            applied.append(name)
    return img, matrix, applied
