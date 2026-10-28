"""HITL Review Queue service — ingest, คิวตรวจ, ยืนยัน/แก้ไข/ปฏิเสธ, Audit Trail, Feedback Data

ทุกการเปลี่ยนแปลงค่าบันทึก AuditLog (Who, When, Original Value, Corrected Value, Reason)
ทุกฟิลด์ที่มนุษย์ตรวจแล้วบันทึก FeedbackSample (ภาพ crop + ข้อความที่ถูกต้อง) สำหรับ fine-tune
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.extraction.kie import FIELDS, FieldPrediction
from app.models import AuditAction, AuditLog, Document, DocumentStatus, ExtractedField, FeedbackSample
from app.pipeline import ExpiryOCRPipeline, PipelineResult
from app.recognition.detector import crop_polygon
from app.rules.validators import validate

REVIEWED = (DocumentStatus.HUMAN_APPROVED, DocumentStatus.HUMAN_CORRECTED, DocumentStatus.REJECTED)


class ReviewError(Exception):
    def __init__(self, message: str, details: list | None = None, status_code: int = 400):
        super().__init__(message)
        self.details = details or []
        self.status_code = status_code


@dataclass
class FieldCorrection:
    name: str
    value: str | None              # ค่า normalize (ISO date / lot) ที่ถูกต้อง
    raw_text: str | None = None    # ข้อความตามที่พิมพ์บนฉลากจริง (ใช้เป็น label สำหรับ fine-tune OCR)
    reason: str | None = None


# ---------------------------------------------------------------- ingest
def _save(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def ingest(session: Session, raw: bytes, filename: str, pipeline: ExpiryOCRPipeline) -> tuple[Document, PipelineResult]:
    settings.ensure_dirs()
    result = pipeline.run(raw)
    ext = Path(filename).suffix.lower() or ".png"
    original_path = _save(settings.storage_dir / "original" / f"{result.sha256[:16]}{ext}", raw)
    ok, buf = cv2.imencode(".png", result.preprocess.image)
    processed_path = _save(settings.storage_dir / "processed" / f"{result.sha256[:16]}.png", buf.tobytes()) if ok else None

    doc = Document(
        filename=filename, sha256=result.sha256, original_path=original_path, processed_path=processed_path,
        status=result.decision.status, overall_confidence=result.overall_confidence, engine=result.engine,
        route_reasons=result.decision.reasons, rule_results=[r.to_dict() for r in result.rules],
        preprocessing={"steps": result.preprocess.steps, "metrics": result.preprocess.metrics,
                       "passes": result.passes, "pass_fields": result.pass_fields},
        ocr_lines=[ln.to_dict() for ln in result.lines], processing_ms=result.processing_ms,
    )
    for name in FIELDS:
        f = result.fields.get(name)
        if f is None:
            continue
        doc.fields.append(ExtractedField(
            name=name, predicted_raw=f.raw_text, predicted_value=f.value,
            final_value=f.value if result.decision.auto_approved else None,
            confidence=f.confidence, date_format=f.date_format, bbox=f.polygon,
        ))
    auto = result.decision.auto_approved
    doc.audit_logs.append(AuditLog(
        action=AuditAction.AUTO_APPROVE if auto else AuditAction.ROUTE_TO_REVIEW, actor="system",
        reason="ผ่านเกณฑ์ความมั่นใจและกฎธุรกิจทุกข้อ" if auto else "; ".join(result.decision.reasons),
    ))
    session.add(doc)
    session.commit()
    return doc, result


# ---------------------------------------------------------------- queries
def get_document(session: Session, doc_id: int) -> Document:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise ReviewError(f"ไม่พบเอกสาร id={doc_id}", status_code=404)
    return doc


def list_documents(session: Session, status: DocumentStatus | None = None, limit: int = 100, offset: int = 0):
    q = select(Document).order_by(Document.id.desc()).limit(limit).offset(offset)
    if status:
        q = q.where(Document.status == status)
    return list(session.scalars(q))


def review_queue(session: Session, limit: int = 50, order: str = "fifo") -> list[Document]:
    """คิว HITL: fifo = เข้าก่อนตรวจก่อน, hardest = ความมั่นใจต่ำสุดก่อน"""
    q = select(Document).where(Document.status == DocumentStatus.PENDING_REVIEW)
    q = q.order_by(Document.overall_confidence.asc() if order == "hardest" else Document.created_at.asc())
    return list(session.scalars(q.limit(limit)))


# ---------------------------------------------------------------- review actions
def _as_prediction(name: str, value: str | None) -> FieldPrediction | None:
    if value is None or value == "":
        return None
    if name == "lot_number":
        return FieldPrediction(name=name, value=value.strip().upper(), raw_text=value, confidence=1.0,
                               char_confidences=[1.0])
    try:
        d = date.fromisoformat(value)
    except ValueError:
        raise ReviewError(f"{name}: ต้องเป็นวันที่รูปแบบ YYYY-MM-DD (ได้ {value!r})", status_code=422)
    return FieldPrediction(name=name, value=d.isoformat(), raw_text=value, confidence=1.0, char_confidences=[1.0],
                           date_format="YYYY-MM-DD", components={"day": d.day, "month": d.month, "year": d.year})


def _save_crop(doc: Document, field_name: str, polygon) -> str | None:
    if not polygon:
        return None
    img = cv2.imread(doc.original_path)
    if img is None:
        return None
    crop = crop_polygon(img, polygon)
    if crop.size == 0:
        return None
    path = settings.storage_dir / "feedback_crops" / f"doc{doc.id}_{field_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    return str(path)


def _line_text_for(doc: Document, f: ExtractedField) -> str | None:
    """หาข้อความทั้งบรรทัดที่ฟิลด์นี้อยู่ (label สำหรับ fine-tune TrOCR เป็นระดับบรรทัด)"""
    for ln in doc.ocr_lines or []:
        if f.predicted_raw and f.predicted_raw in ln["text"]:
            return ln["text"]
    return None


def submit_review(session: Session, doc_id: int, reviewer: str, corrections: list[FieldCorrection],
                  force: bool = False) -> Document:
    """ยืนยัน (ค่าเดิม) หรือแก้ไข (ค่าใหม่) ทุกฟิลด์ในครั้งเดียว

    - ฟิลด์ที่ไม่ได้ส่งมา = ยืนยันค่าที่ระบบทำนาย
    - ค่าหลังแก้ต้องผ่านกฎธุรกิจ (severity=error) ไม่เช่นนั้นต้องส่ง force=True พร้อม reason
    """
    if not reviewer.strip():
        raise ReviewError("ต้องระบุชื่อผู้ตรวจ (reviewer)", status_code=422)
    doc = get_document(session, doc_id)
    if doc.status != DocumentStatus.PENDING_REVIEW and not force:
        raise ReviewError(f"เอกสารสถานะ {doc.status.value} ไม่อยู่ในคิวตรวจ", status_code=409)
    by_name = {c.name: c for c in corrections}
    unknown = set(by_name) - set(FIELDS)
    if unknown:
        raise ReviewError(f"ไม่รู้จักฟิลด์: {sorted(unknown)}", status_code=422)

    # ตรวจค่าสุดท้ายด้วยกฎธุรกิจเดียวกับระบบอัตโนมัติ
    final_values = {f.name: f.predicted_value for f in doc.fields}
    final_values.update({c.name: (c.value or None) for c in corrections})
    preds = {n: p for n, v in final_values.items() if (p := _as_prediction(n, v)) is not None}
    failed = [r.to_dict() for r in validate(preds, max_shelf_life_years=settings.max_shelf_life_years)
              if not r.passed and r.severity == "error"]
    if failed and not force:
        raise ReviewError("ค่าที่ตรวจแล้วยังไม่ผ่านกฎธุรกิจ (ส่ง force=true พร้อมเหตุผลเพื่อยืนยัน)", failed, 422)

    now = datetime.now(timezone.utc)
    any_edit = False
    for name in FIELDS:
        f = doc.field(name)
        corr = by_name.get(name)
        new_value = preds[name].value if name in preds else None
        if f is None and new_value is None:
            continue
        if f is None:   # ระบบอ่านไม่เจอ แต่มนุษย์เติมให้
            f = ExtractedField(name=name, predicted_raw=None, predicted_value=None, confidence=0.0)
            doc.fields.append(f)
        changed = (f.predicted_value or None) != new_value
        any_edit |= changed
        f.final_value = new_value
        f.is_corrected = changed
        doc.audit_logs.append(AuditLog(
            field_name=name, action=AuditAction.EDIT if changed else AuditAction.CONFIRM, actor=reviewer,
            original_value=f.predicted_value, corrected_value=new_value, created_at=now,
            reason=(corr.reason if corr else None) or ("แก้ไขค่าที่ OCR อ่านผิด" if changed else "ยืนยันค่าที่ระบบอ่าน"),
        ))
        line_text = _line_text_for(doc, f)
        corrected_line = line_text
        if corr and corr.raw_text and line_text and f.predicted_raw:
            corrected_line = line_text.replace(f.predicted_raw, corr.raw_text, 1)
        session.add(FeedbackSample(
            document_id=doc.id, field_name=name, image_path=doc.original_path,
            crop_path=_save_crop(doc, name, f.bbox), bbox=f.bbox,
            predicted_text=line_text or f.predicted_raw,
            corrected_text=corrected_line or (corr.raw_text if corr else f.predicted_raw),
            corrected_value=new_value, was_error=changed,
        ))
    if failed and force:
        doc.audit_logs.append(AuditLog(action=AuditAction.CONFIRM, actor=reviewer, created_at=now,
                                       reason="FORCE: ยืนยันแม้ไม่ผ่านกฎ " + ",".join(r["rule_id"] for r in failed)))
    doc.status = DocumentStatus.HUMAN_CORRECTED if any_edit else DocumentStatus.HUMAN_APPROVED
    doc.reviewed_by, doc.reviewed_at = reviewer, now
    session.commit()
    return doc


def reject(session: Session, doc_id: int, reviewer: str, reason: str) -> Document:
    doc = get_document(session, doc_id)
    if not reason.strip():
        raise ReviewError("ต้องระบุเหตุผลการปฏิเสธ", status_code=422)
    doc.status = DocumentStatus.REJECTED
    doc.reviewed_by, doc.reviewed_at = reviewer, datetime.now(timezone.utc)
    doc.audit_logs.append(AuditLog(action=AuditAction.REJECT, actor=reviewer, reason=reason))
    session.commit()
    return doc


# ---------------------------------------------------------------- operational metrics
def operational_metrics(session: Session) -> dict:
    counts = dict(session.execute(select(Document.status, func.count()).group_by(Document.status)).all())
    total = sum(counts.values())
    auto = counts.get(DocumentStatus.AUTO_APPROVED, 0)
    corrected = counts.get(DocumentStatus.HUMAN_CORRECTED, 0)
    approved = counts.get(DocumentStatus.HUMAN_APPROVED, 0)
    reviewed_fields = session.scalar(select(func.count()).select_from(FeedbackSample)) or 0
    error_fields = session.scalar(select(func.count()).select_from(FeedbackSample).where(FeedbackSample.was_error)) or 0
    per_field = {
        name: {"reviewed": int(n), "corrected": int(c or 0)}
        for name, n, c in session.execute(
            select(FeedbackSample.field_name, func.count(), func.sum(FeedbackSample.was_error.cast(Integer)))
            .group_by(FeedbackSample.field_name)).all()
    }
    return {
        "total_documents": total,
        "by_status": {s.value: int(counts.get(s, 0)) for s in DocumentStatus},
        "first_pass_yield": round(auto / total, 4) if total else None,
        "exception_rate": round((total - auto) / total, 4) if total else None,
        "human_correction_rate": round(corrected / (corrected + approved), 4) if corrected + approved else None,
        "field_correction_rate": round(error_fields / reviewed_fields, 4) if reviewed_fields else None,
        "per_field": per_field,
        "avg_confidence": round(float(session.scalar(select(func.avg(Document.overall_confidence))) or 0), 4),
    }

