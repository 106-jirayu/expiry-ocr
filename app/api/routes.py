"""REST API v1"""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from app.alerts import notifier
from app.alerts import service as alerts
from app.config import settings
from app.db import get_session
from app.downstream.exporter import document_json, export_feedback_dataset
from app.downstream.visualize import draw_annotations, encode_png
from app.models import DocumentStatus
from app.pipeline import ExpiryOCRPipeline
from app.review import service
from app.review.service import FieldCorrection
from app.schemas import AckIn, RejectIn, ReviewIn

router = APIRouter(prefix="/api/v1")
MAX_UPLOAD = 15 * 1024 * 1024
_pipeline: ExpiryOCRPipeline | None = None


def get_pipeline() -> ExpiryOCRPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ExpiryOCRPipeline()
    return _pipeline


# ------------------------------------------------------------ 1) Intake
@router.post("/documents", summary="อัปโหลดภาพบรรจุภัณฑ์ -> ประมวลผล OCR -> routing")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_session),
                          pipeline: ExpiryOCRPipeline = Depends(get_pipeline)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "ไฟล์ว่าง")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "ไฟล์ใหญ่เกิน 15MB")
    try:
        doc, _ = service.ingest(db, raw, file.filename or "upload.jpg", pipeline)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return document_json(doc)


@router.get("/documents")
def list_documents(status: DocumentStatus | None = None, limit: int = Query(100, le=500), offset: int = 0,
                   db: Session = Depends(get_session)):
    return [document_json(d, include_audit=False) for d in service.list_documents(db, status, limit, offset)]


@router.get("/documents/{doc_id}")
def get_document(doc_id: int, db: Session = Depends(get_session)):
    return document_json(service.get_document(db, doc_id))


@router.get("/documents/{doc_id}/image", summary="ภาพ original | processed | annotated (มีกรอบไฮไลท์)")
def get_image(doc_id: int, variant: str = Query("annotated", pattern="^(original|processed|annotated)$"),
              db: Session = Depends(get_session)):
    doc = service.get_document(db, doc_id)
    if variant == "processed" and doc.processed_path:
        return Response(Path(doc.processed_path).read_bytes(), media_type="image/png")
    img = cv2.imread(doc.original_path)
    if img is None:
        raise HTTPException(404, "ไม่พบไฟล์ภาพ")
    if variant == "annotated":
        fields = [{"name": f.name, "value": f.final_value or f.predicted_value, "confidence": f.confidence,
                   "polygon": f.bbox} for f in doc.fields]
        img = draw_annotations(img, fields, settings.confidence_threshold, doc.ocr_lines)
    return Response(encode_png(img), media_type="image/png")


@router.get("/documents/{doc_id}/audit", summary="Audit Trail ของเอกสาร")
def get_audit(doc_id: int, db: Session = Depends(get_session)):
    return document_json(service.get_document(db, doc_id))["audit_trail"]


SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "images"


@router.get("/samples", summary="รายชื่อภาพตัวอย่างสำหรับทดลอง")
def list_samples():
    return sorted(p.name for p in SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []


@router.get("/samples/{name}", include_in_schema=False)
def get_sample(name: str):
    path = SAMPLES_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "ไม่พบภาพตัวอย่าง")
    return Response(path.read_bytes(), media_type="image/png")


# ------------------------------------------------------------ 4) HITL Review Queue
@router.get("/review/queue")
def review_queue(limit: int = Query(50, le=500), order: str = Query("fifo", pattern="^(fifo|hardest)$"),
                 db: Session = Depends(get_session)):
    return [document_json(d, include_audit=False) for d in service.review_queue(db, limit, order)]


@router.post("/review/{doc_id}/submit", summary="ยืนยัน (Confirm) / แก้ไข (Edit) ฟิลด์")
def submit_review(doc_id: int, body: ReviewIn, db: Session = Depends(get_session)):
    doc = service.submit_review(db, doc_id, body.reviewer,
                                [FieldCorrection(**c.model_dump()) for c in body.corrections], body.force)
    return document_json(doc)


@router.post("/review/{doc_id}/reject")
def reject(doc_id: int, body: RejectIn, db: Session = Depends(get_session)):
    return document_json(service.reject(db, doc_id, body.reviewer, body.reason))


# ------------------------------------------------------------ 5) Downstream & Feedback
@router.get("/export/documents", summary="ส่งออก JSON ของเอกสารที่ผ่านการอนุมัติแล้ว (downstream)")
def export_documents(status: DocumentStatus | None = None, db: Session = Depends(get_session)):
    docs = service.list_documents(db, status, limit=100000)
    payload = [document_json(d) for d in docs if status or d.status.value in
               ("AUTO_APPROVED", "HUMAN_APPROVED", "HUMAN_CORRECTED")]
    return JSONResponse(payload, headers={"Content-Disposition": "attachment; filename=expiry_export.json"})


@router.get("/export/feedback", summary="ดาวน์โหลด Feedback dataset (zip: feedback.jsonl + crops)")
def export_feedback(only_errors: bool = False, db: Session = Depends(get_session)):
    with tempfile.TemporaryDirectory() as tmp:
        info = export_feedback_dataset(db, Path(tmp), only_errors=only_errors)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in Path(tmp).rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(tmp))
            z.writestr("summary.json", json.dumps(info, ensure_ascii=False, indent=2))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=feedback_dataset.zip"})


@router.get("/metrics", summary="ตัวชี้วัดการทำงานจริง: First-Pass Yield, Exception Rate, Correction Rate")
def metrics(db: Session = Depends(get_session)):
    return service.operational_metrics(db)


# ------------------------------------------------------------ แจ้งเตือนวันหมดอายุ
@router.get("/alerts", summary="รายการแจ้งเตือนวันหมดอายุ (ใกล้หมด / หมดวันนี้ / หมดแล้ว)")
def list_alerts(include_acknowledged: bool = False, db: Session = Depends(get_session)):
    return [alerts.alert_json(a) for a in alerts.list_alerts(db, only_open=not include_acknowledged)]


@router.post("/alerts/check", summary="ตรวจวันหมดอายุทุกเอกสารเดี๋ยวนี้ (ปกติระบบตรวจเองตามรอบ)")
def check_alerts(db: Session = Depends(get_session)):
    created = alerts.scan(db)
    return {"created": [alerts.alert_json(a) for a in created], "channels": notifier.configured_channels()}


@router.post("/alerts/{alert_id}/ack", summary="รับทราบแจ้งเตือน (เช่น นำสินค้าออกจากชั้นแล้ว)")
def ack_alert(alert_id: int, body: AckIn, db: Session = Depends(get_session)):
    a = alerts.acknowledge(db, alert_id, body.user)
    if a is None:
        raise HTTPException(404, "ไม่พบแจ้งเตือน")
    return alerts.alert_json(a)


@router.post("/alerts/test", summary="ส่งข้อความทดสอบไปทุกช่องทางที่ตั้งค่าไว้")
def test_notify():
    channels = notifier.configured_channels()
    sent = notifier.send("✅ ทดสอบแจ้งเตือน Expiry OCR", "ถ้าเห็นข้อความนี้ แปลว่าตั้งค่าช่องทางแจ้งเตือนถูกต้องแล้ว")
    return {"configured": channels, "sent": sent}
