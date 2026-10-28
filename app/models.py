"""ORM models — ตารางหลักของระบบ

documents          : ภาพแต่ละภาพที่เข้าระบบ + สถานะ routing
extracted_fields   : ฟิลด์ที่สกัดได้ (EXP / MFG / LOT) ค่าทำนาย, ค่าสุดท้าย, ความมั่นใจ, กรอบ
audit_logs         : Audit Trail ทุกการกระทำ (ใคร, เมื่อไร, ค่าเดิม, ค่าใหม่, เหตุผล)
feedback_samples   : คู่ (ภาพ crop, ข้อความที่ถูกต้อง) สำหรับ fine-tune โมเดล
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, enum.Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    PENDING_REVIEW = "PENDING_REVIEW"      # อยู่ใน HITL Review Queue
    HUMAN_APPROVED = "HUMAN_APPROVED"      # มนุษย์ยืนยันโดยไม่แก้
    HUMAN_CORRECTED = "HUMAN_CORRECTED"    # มนุษย์แก้ไขอย่างน้อย 1 ฟิลด์
    REJECTED = "REJECTED"                  # ภาพใช้ไม่ได้ / ไม่มีวันหมดอายุ


class AuditAction(str, enum.Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    ROUTE_TO_REVIEW = "ROUTE_TO_REVIEW"
    CONFIRM = "CONFIRM"
    EDIT = "EDIT"
    REJECT = "REJECT"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    original_path: Mapped[str] = mapped_column(String(512))
    processed_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), index=True)
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(64))
    route_reasons: Mapped[list] = mapped_column(JSON, default=list)
    rule_results: Mapped[list] = mapped_column(JSON, default=list)
    preprocessing: Mapped[dict] = mapped_column(JSON, default=dict)
    ocr_lines: Mapped[list] = mapped_column(JSON, default=list)
    processing_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    fields: Mapped[list[ExtractedField]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="ExtractedField.id"
    )
    audit_logs: Mapped[list[AuditLog]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="AuditLog.id"
    )

    def field(self, name: str) -> ExtractedField | None:
        return next((f for f in self.fields if f.name == name), None)


class ExtractedField(Base):
    __tablename__ = "extracted_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(32))               # exp_date | mfg_date | lot_number
    predicted_raw: Mapped[str | None] = mapped_column(String(255), nullable=True)   # ข้อความดิบจาก OCR
    predicted_value: Mapped[str | None] = mapped_column(String(255), nullable=True) # ค่า normalize (ISO date)
    final_value: Mapped[str | None] = mapped_column(String(255), nullable=True)     # ค่าหลังตรวจ
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    date_format: Mapped[str | None] = mapped_column(String(32), nullable=True)
    bbox: Mapped[list | None] = mapped_column(JSON, nullable=True)  # polygon [[x,y]*4] พิกัดภาพต้นฉบับ
    is_corrected: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped[Document] = relationship(back_populates="fields")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    field_name: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction))
    original_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str] = mapped_column(String(128))                # ผู้แก้ไข (หรือ "system")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="audit_logs")


class FeedbackSample(Base):
    __tablename__ = "feedback_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    field_name: Mapped[str] = mapped_column(String(32))
    image_path: Mapped[str] = mapped_column(String(512))           # ภาพต้นฉบับ
    crop_path: Mapped[str | None] = mapped_column(String(512), nullable=True)  # ภาพเฉพาะบรรทัด (สำหรับ TrOCR)
    bbox: Mapped[list | None] = mapped_column(JSON, nullable=True)
    predicted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # ข้อความบรรทัดที่ถูกต้อง (label)
    corrected_value: Mapped[str | None] = mapped_column(String(255), nullable=True)  # ค่า normalize ที่ถูกต้อง
    was_error: Mapped[bool] = mapped_column(Boolean, default=True)
    used_in_training: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
