"""แจ้งเตือนวันหมดอายุ

- คำนวณสถานะจาก "ค่าสุดท้าย" ของ EXP (เฉพาะเอกสารที่อนุมัติแล้ว: อัตโนมัติ หรือคนตรวจแล้ว)
    EXPIRED        : EXP < วันนี้
    EXPIRES_TODAY  : EXP = วันนี้
    EXPIRING_SOON  : เหลือ <= ALERT_DAYS_BEFORE วัน
- สร้าง ExpiryAlert ครั้งเดียวต่อ (เอกสาร, ระดับ) แล้วส่งไปทุกช่องทางที่ตั้งค่า (notifier.py)
- ตรวจทันทีเมื่อเอกสารถูกอนุมัติ และตรวจซ้ำตามรอบ ALERT_CHECK_MINUTES (ภาพที่สแกนไว้ก่อน
  จะถูกแจ้งเตือนเมื่อถึงกำหนด)
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.alerts import notifier
from app.config import settings
from app.models import AlertKind, Document, DocumentStatus, ExpiryAlert, ExtractedField

FINAL = (DocumentStatus.AUTO_APPROVED, DocumentStatus.HUMAN_APPROVED, DocumentStatus.HUMAN_CORRECTED)
TITLE = {
    AlertKind.EXPIRED: "⛔ สินค้าหมดอายุแล้ว",
    AlertKind.EXPIRES_TODAY: "🚨 สินค้าหมดอายุวันนี้",
    AlertKind.EXPIRING_SOON: "⏰ สินค้าใกล้หมดอายุ",
}


def final_exp(doc: Document) -> date | None:
    if doc.status not in FINAL:
        return None
    f = doc.field("exp_date")
    if f is None or not f.final_value:
        return None
    try:
        return date.fromisoformat(f.final_value)
    except ValueError:
        return None


def classify(exp: date, today: date, days_before: int) -> AlertKind | None:
    if exp < today:
        return AlertKind.EXPIRED
    if exp == today:
        return AlertKind.EXPIRES_TODAY
    if (exp - today).days <= days_before:
        return AlertKind.EXPIRING_SOON
    return None


def expiry_info(doc: Document, today: date | None = None, days_before: int | None = None) -> dict | None:
    """สรุปสถานะวันหมดอายุสำหรับแสดงผล (ใส่ใน JSON ของเอกสาร)"""
    exp = final_exp(doc)
    if exp is None:
        return None
    today = today or date.today()
    kind = classify(exp, today, settings.alert_days_before if days_before is None else days_before)
    return {"exp_date": exp.isoformat(), "days_left": (exp - today).days, "status": kind.value if kind else "OK"}


def _message(doc: Document, kind: AlertKind, exp: date, today: date) -> str:
    lot = doc.field("lot_number")
    days = (exp - today).days
    when = (f"หมดอายุไปแล้ว {-days} วัน" if days < 0 else "หมดอายุวันนี้" if days == 0 else f"เหลืออีก {days} วัน")
    lines = [f"วันหมดอายุ: {exp.strftime('%d/%m/%Y')} ({when})",
             f"Lot: {lot.final_value if lot and lot.final_value else '-'}",
             f"เอกสาร #{doc.id} · ไฟล์ {doc.filename}"]
    return "\n".join(lines)


def check_document(session: Session, doc: Document, today: date | None = None, send: bool = True) -> list[ExpiryAlert]:
    today = today or date.today()
    exp = final_exp(doc)
    if exp is None:
        return []
    kind = classify(exp, today, settings.alert_days_before)
    if kind is None:
        return []
    exists = session.scalar(select(ExpiryAlert.id).where(ExpiryAlert.document_id == doc.id, ExpiryAlert.kind == kind))
    if exists:
        return []
    lot = doc.field("lot_number")
    alert = ExpiryAlert(document_id=doc.id, kind=kind, exp_date=exp.isoformat(),
                        lot_number=lot.final_value if lot else None, message=_message(doc, kind, exp, today))
    if send:
        alert.channels_sent = notifier.send(TITLE[kind], alert.message,
                                            {"document_id": doc.id, "kind": kind.value, "exp_date": alert.exp_date,
                                             "lot_number": alert.lot_number})
    session.add(alert)
    session.commit()
    return [alert]


def scan(session: Session, today: date | None = None, send: bool = True) -> list[ExpiryAlert]:
    """ตรวจเอกสารที่อนุมัติแล้วทั้งหมด (เรียกตามรอบเวลา)"""
    created = []
    for doc in session.scalars(select(Document).where(Document.status.in_(FINAL))):
        created += check_document(session, doc, today, send)
    return created


def list_alerts(session: Session, only_open: bool = True, limit: int = 200) -> list[ExpiryAlert]:
    q = select(ExpiryAlert).order_by(ExpiryAlert.exp_date.asc(), ExpiryAlert.id.desc()).limit(limit)
    if only_open:
        q = q.where(ExpiryAlert.acknowledged.is_(False))
    return list(session.scalars(q))


def acknowledge(session: Session, alert_id: int, user: str) -> ExpiryAlert | None:
    a = session.get(ExpiryAlert, alert_id)
    if a is None:
        return None
    a.acknowledged, a.acknowledged_by, a.acknowledged_at = True, user, datetime.now(timezone.utc)
    session.commit()
    return a


def alert_json(a: ExpiryAlert) -> dict:
    return {"id": a.id, "document_id": a.document_id, "kind": a.kind.value, "title": TITLE[a.kind],
            "exp_date": a.exp_date, "days_left": (date.fromisoformat(a.exp_date) - date.today()).days,
            "lot_number": a.lot_number, "message": a.message, "channels_sent": a.channels_sent,
            "acknowledged": a.acknowledged, "acknowledged_by": a.acknowledged_by,
            "created_at": a.created_at.isoformat() if a.created_at else None}


def calendar(session: Session, start: date, end: date, today: date | None = None) -> list[dict]:
    """สินค้าที่อนุมัติแล้วและหมดอายุในช่วง [start, end] สำหรับหน้าปฏิทิน (เรียงตามวันหมดอายุ)"""
    today = today or date.today()
    q = (select(Document, ExtractedField.final_value)
         .join(ExtractedField, ExtractedField.document_id == Document.id)
         .where(Document.status.in_(FINAL), ExtractedField.name == "exp_date",
                ExtractedField.final_value >= start.isoformat(), ExtractedField.final_value <= end.isoformat())
         .order_by(ExtractedField.final_value, Document.id))
    out = []
    for doc, value in session.execute(q):
        try:
            exp = date.fromisoformat(value)
        except ValueError:
            continue
        kind = classify(exp, today, settings.alert_days_before)
        lot, mfg = doc.field("lot_number"), doc.field("mfg_date")
        out.append({"document_id": doc.id, "exp_date": exp.isoformat(), "days_left": (exp - today).days,
                    "status": kind.value if kind else "OK", "doc_status": doc.status.value,
                    "lot_number": lot.final_value if lot else None, "mfg_date": mfg.final_value if mfg else None,
                    "filename": doc.filename})
    return out
