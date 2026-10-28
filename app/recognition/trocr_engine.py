"""TrOCR engine (Transformer-based OCR: ViT encoder + text decoder)

- ตรวจหาบรรทัดข้อความด้วย detector แล้วส่ง crop ทีละบรรทัดเข้า TrOCR (fine-grained recognition)
- คำนวณความมั่นใจรายตัวอักษรจาก softmax probability ของแต่ละ token ที่ decoder เลือก
- โหลดโมเดลครั้งแรกที่ใช้ (lazy) — ต้องติดตั้ง requirements-ml.txt
- ใช้โมเดลที่ fine-tune จาก feedback data ได้โดยตั้ง TROCR_MODEL=<path โฟลเดอร์โมเดล>
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.recognition.base import OCREngine, RecognitionContext, TextLine
from app.recognition.detector import crop_polygon, detect_text_lines


class TrOCREngine(OCREngine):
    name = "trocr"

    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: str | None = None,
                 num_beams: int = 4):
        self.model_name = model_name
        self.device = device
        self.num_beams = num_beams
        self._processor: Any = None
        self._model: Any = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                TrOCRProcessor, VisionEncoderDecoderModel)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("ต้องติดตั้ง requirements-ml.txt ก่อนใช้ TrOCR") from e
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device).eval()

    def recognize_crop(self, crop: np.ndarray) -> tuple[str, list[float]]:
        import torch  # pyright: ignore[reportMissingImports]
        from PIL import Image

        self._load()
        rgb = crop[..., ::-1] if crop.ndim == 3 else np.stack([crop] * 3, -1)
        pixel_values = self._processor(images=Image.fromarray(np.ascontiguousarray(rgb)),
                                       return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            out = self._model.generate(pixel_values, max_new_tokens=48, num_beams=self.num_beams,
                                       output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0]
        # transition scores = log-prob ของ token ที่ถูกเลือกในแต่ละ step (รองรับ beam search)
        scores = self._model.compute_transition_scores(out.sequences, out.scores, out.beam_indices
                                                       if self.num_beams > 1 else None, normalize_logits=True)[0]
        tok = self._processor.tokenizer
        gen_ids = seq[-len(scores):]
        text, confs = "", []
        for tid, s in zip(gen_ids.tolist(), scores.tolist()):
            if tid in tok.all_special_ids:
                continue
            piece = tok.decode([tid], skip_special_tokens=True)
            p = float(np.exp(s))
            text += piece
            confs.extend([p] * len(piece))
        # ตัดช่องว่างหัวท้ายพร้อม conf ให้ตรงกัน
        lstrip = len(text) - len(text.lstrip())
        text_s = text.strip()
        confs = confs[lstrip:lstrip + len(text_s)]
        return text_s, confs

    def recognize(self, image: np.ndarray, ctx: RecognitionContext) -> list[TextLine]:
        lines = []
        for poly in detect_text_lines(image):
            crop = crop_polygon(image, poly)
            if crop.size == 0:
                continue
            text, confs = self.recognize_crop(crop)
            if not text:
                continue
            lines.append(TextLine(text=text, confidence=float(np.mean(confs)) if confs else 0.0,
                                  char_confidences=confs, polygon=poly))
        return lines
