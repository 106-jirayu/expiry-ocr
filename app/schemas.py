from __future__ import annotations

from pydantic import BaseModel, Field


class FieldCorrectionIn(BaseModel):
    name: str = Field(..., examples=["exp_date"])
    value: str | None = Field(None, description="ค่าที่ถูกต้อง: วันที่ YYYY-MM-DD หรือ Lot number")
    raw_text: str | None = Field(None, description="ข้อความตามที่พิมพ์บนฉลาก (สำหรับ fine-tune OCR)")
    reason: str | None = None


class ReviewIn(BaseModel):
    reviewer: str = Field(..., min_length=1, examples=["qa_somchai"])
    corrections: list[FieldCorrectionIn] = Field(default_factory=list,
                                                 description="ฟิลด์ที่ไม่ได้ส่ง = ยืนยันค่าที่ระบบอ่าน")
    force: bool = False


class RejectIn(BaseModel):
    reviewer: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
