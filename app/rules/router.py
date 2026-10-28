"""Routing Logic

AUTO_APPROVED   : ฟิลด์ที่พบทุกตัวมี confidence >= threshold (ค่าเริ่มต้น 0.90)
                  และ ผ่านกฎ severity=error ทุกข้อ
PENDING_REVIEW  : กรณีอื่นทั้งหมด -> HITL Review Queue พร้อมเหตุผล (route_reasons)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.extraction.kie import FieldPrediction
from app.models import DocumentStatus
from app.rules.validators import RuleResult


@dataclass
class RoutingDecision:
    status: DocumentStatus
    reasons: list[str] = field(default_factory=list)

    @property
    def auto_approved(self) -> bool:
        return self.status == DocumentStatus.AUTO_APPROVED


def route(fields: dict[str, FieldPrediction], rules: list[RuleResult], threshold: float = 0.90) -> RoutingDecision:
    reasons: list[str] = []
    for f in fields.values():
        if f.confidence < threshold:
            reasons.append(f"LOW_CONFIDENCE:{f.name}={f.confidence:.2%} < {threshold:.0%}")
        if f.agreement < 1.0:
            reasons.append(f"ENSEMBLE_DISAGREE:{f.name} agreement={f.agreement:.0%}")
    for r in rules:
        if not r.passed and r.severity == "error":
            reasons.append(f"RULE_FAIL:{r.rule_id}:{r.name}:{r.field}")
    status = DocumentStatus.AUTO_APPROVED if not reasons else DocumentStatus.PENDING_REVIEW
    return RoutingDecision(status=status, reasons=reasons)
