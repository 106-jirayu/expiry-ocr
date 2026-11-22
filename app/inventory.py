"""คลังสินค้า (v4): สินค้าที่อนุมัติแล้วพร้อมวันหมดอายุ — ค้นหา กรอง ตั้งชื่อ นำออกจากชั้น ส่งออก CSV

กลุ่มวันหมดอายุ (bucket) ของแต่ละรายการ
    expired : หมดอายุแล้ว (เลยวันหมดอายุ)
    today   : หมดอายุวันนี้
    soon    : เหลือ 1..ALERT_DAYS_BEFORE วัน (ตรงกับเกณฑ์แจ้งเตือน)
    month   : เหลือไม่เกิน 30 วัน
    later   : มากกว่า 30 วัน
สินค้าที่ "นำออกจากชั้นแล้ว" (ขายหมด / ทิ้งแล้ว) จะไม่นับในคลังและไม่แจ้งเตือนอีก แต่ยังค้นดูย้อนหลังได้
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.alerts.service import FINAL
from app.config import settings
from app.models import AuditAction, AuditLog, Document, ExpiryAlert
from app.review.service import ReviewError, get_document

BUCKETS = ("expired", "today", "soon", "month", "later")
BUCKET_TH = {"expired": "หมดอายุแล้ว", "today": "หมดอายุวันนี้", "soon": "ใกล้หมดอายุ",
             "month": "ภายใน 30 วัน", "later": "มากกว่า 30 วัน", "removed": "นำออกจากชั้นแล้ว"}


def bucket_of(days_left: int, days_before: int | None = None) -> str:
    days_before = settings.alert_days_before if days_before is None else days_before
    if days_left < 0:
        return "expired"
    if days_left == 0:
        return "today"
    if days_left <= days_before:
        return "soon"
    if days_left <= 30:
        return "month"
    return "later"


def _value(doc: Document, name: str) -> str | None:
    f = doc.field(name)
    return f.final_value if f else None


def _item(doc: Document, today: date) -> dict | None:
    raw = _value(doc, "exp_date")
    try:
        exp = date.fromisoformat(raw) if raw else None
    except ValueError:
        exp = None
    if exp is None:
        return None
    days = (exp - today).days
    return {
        "document_id": doc.id, "product_name": doc.display_name, "filename": doc.filename, "engine": doc.engine,
        "exp_date": exp.isoformat(), "days_left": days,
        "bucket": "removed" if doc.removed_at else bucket_of(days),
        "mfg_date": _value(doc, "mfg_date"), "lot_number": _value(doc, "lot_number"),
        "doc_status": doc.status.value, "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "removed": None if doc.removed_at is None else
        {"at": doc.removed_at.isoformat(), "by": doc.removed_by, "note": doc.removed_note},
    }


def _matches(item: dict, q: str) -> bool:
    q = q.strip().lower()
    if not q:
        return True
    hay = " ".join(str(item.get(k) or "") for k in ("product_name", "lot_number", "filename", "exp_date"))
    return q.lstrip("#") == str(item["document_id"]) or q in hay.lower()


def inventory(session: Session, q: str = "", bucket: str | None = None, sort: str = "exp",
              today: date | None = None) -> dict:
    """คืน {summary: {bucket: จำนวน}, items: [...]} — summary นับหลังค้นหา (q) แต่ก่อนกรอง bucket"""
    today = today or date.today()
    docs = session.scalars(select(Document).where(Document.status.in_(FINAL)))
    items = [it for d in docs if (it := _item(d, today)) is not None and _matches(it, q)]
    summary = {b: 0 for b in (*BUCKETS, "removed")}
    for it in items:
        summary[it["bucket"]] += 1
    summary["in_stock"] = sum(summary[b] for b in BUCKETS)
    if bucket == "all":
        pass
    elif bucket:
        items = [it for it in items if it["bucket"] == bucket]
    else:  # ค่าเริ่มต้น: เฉพาะที่ยังอยู่บนชั้น
        items = [it for it in items if it["bucket"] != "removed"]
    if sort == "new":
        items.sort(key=lambda it: it["document_id"], reverse=True)
    else:  # ใกล้หมดอายุก่อน
        items.sort(key=lambda it: (it["exp_date"], it["document_id"]))
    return {"today": today.isoformat(), "alert_days_before": settings.alert_days_before,
            "summary": summary, "items": items}


def rename(session: Session, doc_id: int, user: str, product_name: str | None) -> Document:
    doc = get_document(session, doc_id)
    new = (product_name or "").strip() or None
    if new == doc.product_name:
        return doc
    doc.audit_logs.append(AuditLog(field_name="product_name", action=AuditAction.EDIT, actor=user,
                                   original_value=doc.product_name, corrected_value=new, reason="ตั้งชื่อสินค้า"))
    doc.product_name = new
    if doc.engine == "manual" and new:
        doc.filename = new
    session.commit()
    return doc


def remove_from_shelf(session: Session, doc_id: int, user: str, note: str | None = None) -> Document:
    """นำสินค้าออกจากชั้น (ขายหมด / ทิ้ง / คืนซัพพลายเออร์) -> ปิดแจ้งเตือนที่ค้างของรายการนี้"""
    doc = get_document(session, doc_id)
    if doc.status not in FINAL:
        raise ReviewError("ต้องตรวจ/อนุมัติรายการนี้ก่อนจึงจะนำออกจากชั้นได้", status_code=409)
    if doc.removed_at is not None:
        return doc
    now = datetime.now(timezone.utc)
    doc.removed_at, doc.removed_by, doc.removed_note = now, user, (note or "").strip() or None
    doc.audit_logs.append(AuditLog(field_name="shelf", action=AuditAction.EDIT, actor=user, original_value="on_shelf",
                                   corrected_value="removed", reason=doc.removed_note or "นำออกจากชั้น"))
    for a in session.scalars(select(ExpiryAlert).where(ExpiryAlert.document_id == doc.id,
                                                       ExpiryAlert.acknowledged.is_(False))):
        a.acknowledged, a.acknowledged_by, a.acknowledged_at = True, user, now
    session.commit()
    return doc


def restore_to_shelf(session: Session, doc_id: int, user: str) -> Document:
    doc = get_document(session, doc_id)
    if doc.removed_at is None:
        return doc
    doc.audit_logs.append(AuditLog(field_name="shelf", action=AuditAction.EDIT, actor=user, original_value="removed",
                                   corrected_value="on_shelf", reason="นำกลับขึ้นชั้น"))
    doc.removed_at = doc.removed_by = doc.removed_note = None
    session.commit()
    return doc


CSV_COLUMNS = [("document_id", "รหัส"), ("product_name", "ชื่อสินค้า"), ("exp_date", "วันหมดอายุ"),
               ("days_left", "เหลือ (วัน)"), ("bucket", "สถานะ"), ("mfg_date", "วันผลิต"), ("lot_number", "Lot"),
               ("doc_status", "การตรวจ"), ("created_at", "บันทึกเมื่อ")]


def to_csv(items: list[dict]) -> bytes:
    """CSV แบบ UTF-8 มี BOM เพื่อให้ Excel เปิดภาษาไทยได้ถูกต้อง"""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([th for _, th in CSV_COLUMNS])
    for it in items:
        row = {**it, "bucket": BUCKET_TH.get(it["bucket"], it["bucket"]),
               "product_name": it["product_name"] or it["filename"]}
        w.writerow(["" if row.get(k) is None else row[k] for k, _ in CSV_COLUMNS])
    return ("﻿" + buf.getvalue()).encode("utf-8")
