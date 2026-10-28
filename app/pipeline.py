"""End-to-end pipeline: Intake -> Recognition -> KIE -> Confidence & Rules -> Routing

ไม่ผูกกับฐานข้อมูล (ใช้ได้ทั้งใน API และสคริปต์ประเมินผล)
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from app.config import settings
from app.extraction.consensus import merge_passes
from app.extraction.kie import FieldPrediction, extract_fields, to_structured_json, unclaimed_dates
from app.intake.preprocessing import PreprocessConfig, PreprocessResult, decode_image, preprocess
from app.recognition.base import OCREngine, RecognitionContext, TextLine
from app.recognition.factory import build_engine
from app.rules.calibration import Calibrator
from app.rules.confidence import apply_confidence, document_confidence
from app.rules.router import RoutingDecision, route
from app.rules.validators import RuleResult, validate

# preprocessing variant ของแต่ละรอบ (multi-pass) — ให้ engine เห็นภาพในมุมที่ต่างกันเล็กน้อย
PASS_VARIANTS: list[dict] = [
    {},                                                    # ครบทุกขั้นตอน
    {"deskew": False},                                     # ไม่หมุนภาพ (กันกรณีประมาณมุมเอียงผิด)
    {"denoise": False, "deblur": False, "contrast": False, "deskew": False},  # ภาพต้นฉบับ (แค่ปรับขนาด)
]


@dataclass
class PipelineResult:
    sha256: str
    fields: dict[str, FieldPrediction]
    rules: list[RuleResult]
    decision: RoutingDecision
    overall_confidence: float
    preprocess: PreprocessResult
    lines: list[TextLine]                 # บรรทัดจากรอบแรก (พิกัดภาพต้นฉบับ)
    engine: str
    processing_ms: float
    passes: int = 1
    pass_fields: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "sha256": self.sha256,
            "status": self.decision.status.value,
            "overall_confidence": self.overall_confidence,
            "route_reasons": self.decision.reasons,
            "fields": to_structured_json(self.fields),
            "rules": [r.to_dict() for r in self.rules],
            "preprocessing": {"steps": self.preprocess.steps, "metrics": self.preprocess.metrics},
            "engine": self.engine,
            "passes": self.passes,
            "processing_ms": round(self.processing_ms, 1),
        }


class ExpiryOCRPipeline:
    def __init__(self, engine: OCREngine | None = None, threshold: float | None = None,
                 passes: int | None = None, preprocess_config: PreprocessConfig | None = None,
                 today: date | None = None, max_shelf_life_years: int | None = None,
                 calibrator: Calibrator | None = None):
        self.engine = engine or build_engine()
        self.threshold = settings.confidence_threshold if threshold is None else threshold
        self.passes = max(1, min(len(PASS_VARIANTS), passes or settings.ensemble_passes))
        self.base_config = preprocess_config or PreprocessConfig()
        self.today = today
        self.max_shelf_life_years = max_shelf_life_years or settings.max_shelf_life_years
        self.calibrator = calibrator if calibrator is not None else Calibrator(settings.calibration_file)

    def _read_pass(self, original: np.ndarray, raw: bytes, sha: str, i: int):
        cfg = replace(self.base_config, **PASS_VARIANTS[i])
        pre = preprocess(original, cfg)
        ctx = RecognitionContext(raw_bytes=raw, sha256=sha, matrix=pre.matrix, original_image=original, pass_index=i)
        lines = self.engine.recognize(pre.image, ctx)
        for ln in lines:  # เก็บพิกัดเป็นของภาพต้นฉบับเสมอ เพื่อไฮไลท์ใน UI
            if ln.polygon is not None:
                ln.polygon = pre.to_original(ln.polygon).round(1).tolist()
        return pre, lines

    def run(self, raw: bytes) -> PipelineResult:
        t0 = time.perf_counter()
        sha = hashlib.sha256(raw).hexdigest()
        original = decode_image(raw)
        first_pre, first_lines, pass_fields = None, [], []
        for i in range(self.passes):
            pre, lines = self._read_pass(original, raw, sha, i)
            if i == 0:
                first_pre, first_lines = pre, lines
            pass_fields.append(extract_fields(lines, self.today))
        assert first_pre is not None
        snapshot = [{k: v.value for k, v in p.items()} for p in pass_fields]
        engine_name = getattr(self.engine, "last_used", self.engine.name)
        fields = self.calibrator.apply(apply_confidence(merge_passes(pass_fields)), engine_name)
        rules = validate(fields, self.today, self.max_shelf_life_years,
                         extra_dates=unclaimed_dates(first_lines, fields, self.today))
        decision = route(fields, rules, self.threshold)
        return PipelineResult(
            sha256=sha, fields=fields, rules=rules, decision=decision,
            overall_confidence=document_confidence(fields), preprocess=first_pre, lines=first_lines,
            engine=engine_name, processing_ms=(time.perf_counter() - t0) * 1000,
            passes=self.passes, pass_fields=snapshot,
        )
