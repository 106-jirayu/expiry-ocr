import cv2
import numpy as np

from app.intake.augmentation import augment, dilation, erosion, gaussian_blur, random_rotation
from app.intake.preprocessing import blur_score, estimate_skew, preprocess, rotate


def make_label(w=600, h=200):
    img = np.full((h, w, 3), 245, np.uint8)
    for i, t in enumerate(["MFG 01/06/2026", "EXP 01/06/2027"]):
        cv2.putText(img, t, (20, 70 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    return img


def test_deskew_recovers_angle():
    img = make_label()
    rotated, _ = rotate(img, 7)
    assert abs(estimate_skew(rotated) - 7) < 2.0


def test_preprocess_upscales_and_maps_points_back():
    img = make_label(300, 100)
    res = preprocess(img)
    assert min(res.image.shape[:2]) >= 600
    pts = np.array([[10, 10], [200, 80]])
    back = res.to_original(res.to_processed(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_deblur_increases_sharpness():
    img = cv2.GaussianBlur(make_label(), (0, 0), 2.5)
    res = preprocess(img)
    assert "deblur" in " ".join(res.steps)
    assert res.metrics["blur_after"] > res.metrics["blur_before"]


def test_augmentations_shapes_and_range():
    rng = np.random.default_rng(0)
    img = make_label()
    out, m, ang = random_rotation(img, rng)
    assert -10 <= ang <= 10 and m.shape == (2, 3)
    for fn in (gaussian_blur, erosion, dilation):
        assert fn(img, rng).shape == img.shape
    assert blur_score(gaussian_blur(img, rng)) < blur_score(img)
    # erosion ทำให้ตัวอักษรบางลง => พิกเซลเข้มน้อยลง, dilation หนาขึ้น
    dark = lambda x: int((cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) < 128).sum())  # noqa: E731
    assert dark(erosion(img, rng)) < dark(img) < dark(dilation(img, rng))
    aug, m, applied = augment(img, rng)
    assert aug.ndim == 3 and isinstance(applied, list)
