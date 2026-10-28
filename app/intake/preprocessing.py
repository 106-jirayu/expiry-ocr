"""Intake Layer — การปรับสภาพภาพก่อนส่งเข้า OCR

ขั้นตอน (ทุกขั้นตอนเปิด/ปิดได้ผ่าน PreprocessConfig):
  1. Resolution   : ขยายภาพเล็กให้ด้านสั้นอย่างน้อย min_side px (ตัวอักษรบนฉลากมักเล็กมาก)
  2. Denoise      : ลด noise ด้วย Non-local Means
  3. Deskew       : หามุมเอียงของบรรทัดข้อความแล้วหมุนกลับ
  4. Contrast     : CLAHE บนช่อง L (LAB) สำหรับภาพคอนทราสต์ต่ำ / แสงไม่สม่ำเสมอ
  5. De-blur      : ตรวจวัดความเบลอ (variance of Laplacian) ถ้าเบลอให้ทำ unsharp mask
                    หรือ Wiener deconvolution

ผลลัพธ์เก็บ affine matrix (original -> processed) เพื่อแปลงพิกัดกรอบกลับไปบนภาพต้นฉบับ
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    min_side: int = 640
    max_side: int = 2000
    denoise: bool = True
    deskew: bool = True
    max_skew_deg: float = 15.0
    contrast: bool = True
    clahe_clip: float = 2.5
    deblur: bool = True
    blur_threshold: float = 120.0   # variance of Laplacian ต่ำกว่านี้ถือว่าเบลอ
    deblur_method: str = "unsharp"  # unsharp | wiener


@dataclass
class PreprocessResult:
    image: np.ndarray                    # ภาพ BGR หลังปรับ
    matrix: np.ndarray                   # 2x3 affine: original -> processed
    steps: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_original(self, points: np.ndarray | list) -> np.ndarray:
        """แปลงจุดจากพิกัดภาพที่ปรับแล้วกลับไปพิกัดภาพต้นฉบับ"""
        inv = cv2.invertAffineTransform(self.matrix)
        return transform_points(points, inv)

    def to_processed(self, points: np.ndarray | list) -> np.ndarray:
        return transform_points(points, self.matrix)


def transform_points(points, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    return (np.hstack([pts, ones]) @ matrix.T).astype(np.float64)


# ---------------------------------------------------------------- quality metrics
def blur_score(image: np.ndarray) -> float:
    """Variance of Laplacian — ยิ่งต่ำยิ่งเบลอ"""
    gray = to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast_score(image: np.ndarray) -> float:
    """RMS contrast (ส่วนเบี่ยงเบนมาตรฐานของความสว่าง 0-255)"""
    return float(to_gray(image).std())


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("ไม่สามารถอ่านไฟล์ภาพได้ (รองรับ JPG/PNG/BMP/WebP)")
    return img


# ---------------------------------------------------------------- steps
def resize_resolution(image: np.ndarray, min_side: int, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if min(h, w) < min_side:
        scale = min_side / min(h, w)
    if max(h, w) * scale > max_side:
        scale = max_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return image, 1.0
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp), scale


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 11)


def estimate_skew(image: np.ndarray, max_deg: float = 15.0) -> float:
    """ประมาณมุมเอียงของข้อความ (องศา, บวก = ทวนเข็ม) จากบรรทัดข้อความที่ถูก dilate รวมเป็นก้อน"""
    gray = to_gray(image)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(bw) > 127:            # ให้ข้อความเป็นสีขาว
        bw = 255 - bw
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 30), 3))
    lines = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles, weights = [], []
    min_area = gray.shape[0] * gray.shape[1] * 0.002
    for c in contours:
        (_, _), (w, h), ang = cv2.minAreaRect(c)
        if w * h < min_area or min(w, h) == 0:
            continue
        if w < h:                     # normalize ให้เป็นมุมของด้านยาว
            ang = ang - 90
            w, h = h, w
        if w / h < 2.5:               # เอาเฉพาะก้อนที่ยาวแบบบรรทัด
            continue
        if ang < -45:
            ang += 90
        if ang > 45:
            ang -= 90
        if abs(ang) <= max_deg:
            angles.append(ang)
            weights.append(w)
    if not angles:
        return 0.0
    # minAreaRect ใช้แกน y ชี้ลง -> กลับเครื่องหมายให้ บวก = ทวนเข็ม
    return float(-np.average(angles, weights=weights))


def rotate(image: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """หมุนภาพรอบจุดกึ่งกลาง (บวก = ทวนเข็ม) ขยายแคนวาสไม่ให้ภาพถูกตัด"""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = tuple(int(v) for v in np.median(image.reshape(-1, image.shape[-1] if image.ndim == 3 else 1), axis=0))
    out = cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=border if image.ndim == 3 else border[0])
    return out, m


def enhance_contrast(image: np.ndarray, clip: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def unsharp_mask(image: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def wiener_deblur(image: np.ndarray, kernel_size: int = 5, snr: float = 0.01) -> np.ndarray:
    """Wiener deconvolution โดยสมมติ PSF เป็น Gaussian (เหมาะกับภาพหลุดโฟกัส)"""
    k = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    psf = k @ k.T
    out = np.zeros_like(image, dtype=np.float64)
    channels = cv2.split(image.astype(np.float64) / 255.0)
    for i, ch in enumerate(channels):
        psf_pad = np.zeros_like(ch)
        psf_pad[:kernel_size, :kernel_size] = psf
        psf_pad = np.roll(psf_pad, (-(kernel_size // 2), -(kernel_size // 2)), axis=(0, 1))
        H = np.fft.fft2(psf_pad)
        G = np.fft.fft2(ch)
        F = np.conj(H) / (np.abs(H) ** 2 + snr) * G
        res = np.real(np.fft.ifft2(F))
        if image.ndim == 3:
            out[..., i] = res
        else:
            out = res
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive threshold — ใช้กับ OCR บางตัวที่ชอบภาพขาวดำ (ไม่ได้อยู่ใน pipeline หลัก)"""
    return cv2.adaptiveThreshold(to_gray(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


# ---------------------------------------------------------------- pipeline
def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> PreprocessResult:
    cfg = config or PreprocessConfig()
    img = to_bgr(image.copy())
    matrix = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    steps: list[str] = []
    metrics = {
        "input_size": [int(img.shape[1]), int(img.shape[0])],
        "blur_before": round(blur_score(img), 2),
        "contrast_before": round(contrast_score(img), 2),
    }

    img, scale = resize_resolution(img, cfg.min_side, cfg.max_side)
    if scale != 1.0:
        matrix = np.array([[scale, 0, 0], [0, scale, 0]]) @ np.vstack([matrix, [0, 0, 1]])
        matrix = matrix[:2]
        steps.append(f"resize x{scale:.2f}")

    if cfg.denoise:
        img = denoise(img)
        steps.append("denoise")

    if cfg.deskew:
        angle = estimate_skew(img, cfg.max_skew_deg)
        metrics["skew_deg"] = round(angle, 2)
        if abs(angle) >= 0.5:
            img, rot = rotate(img, -angle)
            matrix = (np.vstack([rot, [0, 0, 1]]) @ np.vstack([matrix, [0, 0, 1]]))[:2]
            steps.append(f"deskew {-angle:+.1f}°")

    if cfg.contrast and contrast_score(img) < 60:
        img = enhance_contrast(img, cfg.clahe_clip)
        steps.append("clahe")

    if cfg.deblur:
        b = blur_score(img)
        if b < cfg.blur_threshold:
            img = wiener_deblur(img) if cfg.deblur_method == "wiener" else unsharp_mask(img)
            steps.append(f"deblur:{cfg.deblur_method}")

    metrics.update(
        output_size=[int(img.shape[1]), int(img.shape[0])],
        blur_after=round(blur_score(img), 2),
        contrast_after=round(contrast_score(img), 2),
    )
    return PreprocessResult(image=img, matrix=matrix, steps=steps, metrics=metrics)
