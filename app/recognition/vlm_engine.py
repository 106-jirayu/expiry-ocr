"""Vision-Language Model engine ผ่าน OpenAI-compatible Chat Completions API

ใช้ได้กับ VLM ที่เสิร์ฟด้วย vLLM / TGI / LM Studio ฯลฯ เช่น Qwen2.5-VL, InternVL, MiniCPM-V
- ขอให้โมเดลถอดข้อความทุกบรรทัดพร้อม bbox เป็น JSON
- ถ้า server รองรับ logprobs จะใช้คำนวณความมั่นใจรายตัวอักษร
  ถ้าไม่รองรับ ใช้ค่า default_confidence (ต่ำกว่า threshold) => ต้องผ่าน ensemble หรือ HITL
"""
from __future__ import annotations

import base64
import json
import math
import re

import cv2
import httpx
import numpy as np

from app.recognition.base import OCREngine, RecognitionContext, TextLine

PROMPT = (
    "You are a precise OCR system for food/product packaging. Transcribe EVERY line of printed text "
    "exactly as it appears (keep original characters, punctuation, spacing; do not correct or translate). "
    "Pay special attention to expiry (EXP/BB/BBE/USE BY/หมดอายุ), manufacturing (MFG/MFD/ผลิต) dates and LOT codes. "
    'Return ONLY JSON: {"lines":[{"text":"...","bbox":[x0,y0,x1,y1]}]} with bbox in pixel coordinates '
    "of the given image."
)


class VLMEngine(OCREngine):
    name = "vlm"

    def __init__(self, endpoint: str, model: str, api_key: str = "", timeout: float = 60.0,
                 default_confidence: float = 0.85):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.default_confidence = default_confidence

    def _request(self, image: np.ndarray) -> dict:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("encode image failed")
        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": self.model,
            "temperature": 0,
            "logprobs": True,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = httpx.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _parse_json(content: str) -> dict:
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group(0)) if m else {"lines": []}

    def _char_confidences(self, choice: dict, text: str) -> list[float] | None:
        """ประมาณ conf รายตัวอักษรจาก token logprobs ที่ตรงกับข้อความบรรทัดนั้น"""
        tokens = ((choice.get("logprobs") or {}).get("content")) or []
        if not tokens:
            return None
        stream = "".join(t["token"] for t in tokens)
        start = stream.find(text)
        if start < 0:
            return None
        confs, pos = [], 0
        for t in tokens:
            p = math.exp(t["logprob"])
            for _ in t["token"]:
                if start <= pos < start + len(text):
                    confs.append(p)
                pos += 1
        return confs if len(confs) == len(text) else None

    def recognize(self, image: np.ndarray, ctx: RecognitionContext) -> list[TextLine]:
        resp = self._request(image)
        choice = resp["choices"][0]
        data = self._parse_json(choice["message"]["content"])
        lines = []
        for item in data.get("lines", []):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            confs = self._char_confidences(choice, text) or [self.default_confidence] * len(text)
            poly = None
            if isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4:
                x0, y0, x1, y1 = map(float, item["bbox"])
                poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            lines.append(TextLine(text=text, confidence=float(np.mean(confs)), char_confidences=confs, polygon=poly))
        return lines
