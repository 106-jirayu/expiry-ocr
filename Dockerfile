# Deploy ได้ทั้ง Hugging Face Spaces (Docker), Render, Railway, Google Cloud Run
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# ภาพตัวอย่างสำหรับปุ่ม "ลองด้วยภาพตัวอย่าง" + โหลดโมเดล OCR ล่วงหน้า
RUN python -m scripts.generate_synthetic --out data/synthetic --n 40 --seed 2026 \
    && python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" \
    && mkdir -p data/storage models && chmod -R 777 data models

ENV OCR_ENGINE=rapidocr \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
