"""ค่าคอนฟิกของระบบ อ่านจาก environment variables (ดู .env.example)"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """โหลดไฟล์ .env แบบง่าย ๆ (ไม่ทับค่าที่ตั้งไว้แล้วใน environment)"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/expiry_ocr.db"))
    ocr_engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "rapidocr"))
    trocr_model: str = field(default_factory=lambda: os.getenv("TROCR_MODEL", "microsoft/trocr-base-printed"))
    vlm_endpoint: str = field(default_factory=lambda: os.getenv("VLM_ENDPOINT", "http://localhost:8001/v1/chat/completions"))
    vlm_model: str = field(default_factory=lambda: os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    vlm_api_key: str = field(default_factory=lambda: os.getenv("VLM_API_KEY", ""))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.90")))
    ensemble_passes: int = field(default_factory=lambda: int(os.getenv("ENSEMBLE_PASSES", "3")))
    max_shelf_life_years: int = field(default_factory=lambda: int(os.getenv("MAX_SHELF_LIFE_YEARS", "10")))
    calibration_file: Path = field(default_factory=lambda: Path(os.getenv("CALIBRATION_FILE", "./models/calibration.json")))
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", "./data/storage")))

    def ensure_dirs(self) -> None:
        for sub in ("original", "processed", "annotated", "feedback_crops"):
            (self.storage_dir / sub).mkdir(parents=True, exist_ok=True)


settings = Settings()
