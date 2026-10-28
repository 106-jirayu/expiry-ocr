"""Downstream & Feedback Loop

- document_json(): JSON มาตรฐานส่งต่อระบบปลายทาง (ERP / WMS / ระบบคลังสินค้า)
- export_feedback_dataset(): ส่งออกคู่ (ภาพ crop, ข้อความที่ถูกต้อง) เป็น JSONL สำหรับ fine-tune TrOCR/VLM
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.alerts.service import expiry_info
from app.models import Document, DocumentStatus, FeedbackSample

FINAL_STATUSES = {DocumentStatus.AUTO_APPROVED, DocumentStatus.HUMAN_APPROVED, DocumentStatus.HUMAN_CORRECTED}


def _iso(dt):
    return dt.isoformat() if dt else None


def document_json(doc: Document, include_audit: bool = True) -> dict:
    final = doc.status in FINAL_STATUSES
    out = {
        "document_id": doc.id,
        "filename": doc.filename,
        "sha256": doc.sha256,
        "status": doc.status.value,
        "is_final": final,
        "overall_confidence": doc.overall_confidence,
        "expiry": expiry_info(doc),
        "engine": doc.engine,
        "route_reasons": doc.route_reasons,
        "fields": {
            f.name: {
                "value": f.final_value if final else None,
                "predicted_value": f.predicted_value,
                "predicted_raw": f.predicted_raw,
                "confidence": f.confidence,
                "date_format": f.date_format,
                "bbox": f.bbox,
                "is_corrected": f.is_corrected,
            } for f in doc.fields
        },
        "rules": doc.rule_results,
        "preprocessing": doc.preprocessing,
        "processing_ms": doc.processing_ms,
        "created_at": _iso(doc.created_at),
        "reviewed_by": doc.reviewed_by,
        "reviewed_at": _iso(doc.reviewed_at),
    }
    if include_audit:
        out["audit_trail"] = [
            {"who": a.actor, "when": _iso(a.created_at), "action": a.action.value, "field": a.field_name,
             "original_value": a.original_value, "corrected_value": a.corrected_value, "reason": a.reason}
            for a in doc.audit_logs
        ]
    return out


def export_feedback_dataset(session: Session, out_dir: Path, only_errors: bool = False,
                            mark_used: bool = False) -> dict:
    """เขียน out_dir/feedback.jsonl + out_dir/crops/*.png

    แต่ละบรรทัด: {"image": "crops/x.png", "text": "<ข้อความบรรทัดที่ถูกต้อง>", "field": ..., "was_error": ...}
    ใช้ fine-tune TrOCR ได้ทันที (ดู scripts/finetune_trocr.py)
    """
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    q = select(FeedbackSample).order_by(FeedbackSample.id)
    if only_errors:
        q = q.where(FeedbackSample.was_error)
    n = errors = 0
    with open(out_dir / "feedback.jsonl", "w", encoding="utf-8") as fh:
        for s in session.scalars(q):
            if not s.crop_path or not Path(s.crop_path).exists() or not s.corrected_text:
                continue
            dst = out_dir / "crops" / Path(s.crop_path).name
            shutil.copyfile(s.crop_path, dst)
            fh.write(json.dumps({
                "image": f"crops/{dst.name}", "text": s.corrected_text, "predicted": s.predicted_text,
                "field": s.field_name, "value": s.corrected_value, "was_error": s.was_error,
                "document_id": s.document_id, "source_image": s.image_path,
            }, ensure_ascii=False) + "\n")
            n += 1
            errors += int(s.was_error)
            if mark_used:
                s.used_in_training = True
    if mark_used:
        session.commit()
    return {"samples": n, "error_samples": errors, "path": str(out_dir / "feedback.jsonl")}
