"""Fine-tune TrOCR ด้วย Feedback Data จาก HITL (ปิดวงจร Feedback Loop)

ขั้นตอน:
  1. ดาวน์โหลด dataset: GET /api/v1/export/feedback  แล้วแตก zip ไว้ที่ data/feedback
  2. python -m scripts.finetune_trocr --data data/feedback --out models/trocr-expiry --epochs 10
  3. ตั้ง OCR_ENGINE=trocr และ TROCR_MODEL=models/trocr-expiry แล้วรีสตาร์ท API
  4. รัน evaluation/evaluate.py เทียบก่อน-หลัง (CER, Field Accuracy, FPY)

มี data augmentation (rotation ±10°, blur, erosion, dilation) ระหว่างเทรนเพื่อลด overfit กับข้อมูลน้อย
ต้องติดตั้ง requirements-ml.txt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from app.intake.augmentation import AugmentationPolicy, augment


def load_samples(data_dir: Path) -> list[dict]:
    rows = [json.loads(line) for line in open(data_dir / "feedback.jsonl", encoding="utf-8") if line.strip()]
    return [r for r in rows if r.get("text") and (data_dir / r["image"]).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/feedback")
    ap.add_argument("--base", default="microsoft/trocr-base-printed")
    ap.add_argument("--out", default="models/trocr-expiry")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import torch  # pyright: ignore[reportMissingImports]
    from PIL import Image
    from transformers import (  # pyright: ignore[reportMissingImports]
        TrOCRProcessor, VisionEncoderDecoderModel)

    from evaluation.metrics import cer

    random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data_dir = Path(a.data)
    samples = load_samples(data_dir)
    if len(samples) < 10:
        raise SystemExit(f"ข้อมูลน้อยเกินไป ({len(samples)} ตัวอย่าง) — ตรวจงานใน HITL เพิ่มก่อน")
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * a.val_ratio))
    val, train = samples[:n_val], samples[n_val:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(a.base)
    model = VisionEncoderDecoderModel.from_pretrained(a.base).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    policy = AugmentationPolicy(p_rotation=0.5, max_blur_sigma=1.2)

    def to_pil(path: Path, aug: bool):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        if aug:
            img, _, _ = augment(img, rng, policy)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def evaluate():
        model.eval()
        preds = []
        with torch.no_grad():
            for s in val:
                pv = processor(images=to_pil(data_dir / s["image"], False), return_tensors="pt").pixel_values.to(device)
                ids = model.generate(pv, max_new_tokens=48)
                preds.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        model.train()
        return cer(preds, [s["text"] for s in val])

    print(f"train={len(train)} val={len(val)} device={device} | CER ก่อนเทรน = {evaluate():.4f}")
    best = float("inf")
    model.train()
    for epoch in range(a.epochs):
        random.shuffle(train)
        total = 0.0
        for i in range(0, len(train), a.batch):
            batch = train[i:i + a.batch]
            pv = processor(images=[to_pil(data_dir / s["image"], True) for s in batch],
                           return_tensors="pt").pixel_values.to(device)
            labels = processor.tokenizer([s["text"] for s in batch], padding=True, return_tensors="pt").input_ids
            labels[labels == processor.tokenizer.pad_token_id] = -100
            loss = model(pixel_values=pv, labels=labels.to(device)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item() * len(batch)
        val_cer = evaluate()
        print(f"epoch {epoch + 1}: loss={total / len(train):.4f} val_CER={val_cer:.4f}")
        if val_cer < best:
            best = val_cer
            model.save_pretrained(a.out)
            processor.save_pretrained(a.out)
    print(f"บันทึกโมเดลที่ดีที่สุด (val CER={best:.4f}) ที่ {a.out}")


if __name__ == "__main__":
    main()
