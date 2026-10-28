# Intelligent Expiration Date AI OCR System with Human-in-the-Loop Self-Correction

ระบบ AI OCR อ่านวันหมดอายุ (EXP), วันผลิต (MFG) และ Lot Number บนบรรจุภัณฑ์อาหาร/สินค้า
พร้อมคิวตรวจโดยมนุษย์ (HITL), Audit Trail และ Feedback Loop สำหรับปรับปรุงโมเดล — โปรเจคจบ (Capstone Project)

**ไฮไลท์**
- สถาปัตยกรรม 5 ชั้น: Intake → Recognition → Confidence & Rules → HITL Review → Downstream & Feedback
- **AI OCR ตัวจริงรันบน CPU**: RapidOCR (โมเดล PP-OCR ของ PaddleOCR แบบ ONNX) เป็นค่าเริ่มต้น ไม่ต้องมี GPU / API key
- รองรับ OCR อื่นผ่าน Adapter: **TrOCR**, **Vision-Language Model** (OpenAI-compatible API) และ **Mock** (สำหรับเทสต์)
- **Confidence Calibration**: แปลงคะแนนดิบของ OCR ให้เป็นความน่าจะเป็นที่อ่านถูกจริง ก่อนเทียบเกณฑ์ 90%
- **Multi-pass Ensemble**: อ่านภาพ 3 รอบด้วย preprocessing ต่างกัน แล้วลงคะแนนเสียง ถ้าไม่ตรงกัน → ส่งคนตรวจ
- กฎธุรกิจ 9 ข้อ + ความมั่นใจระดับฟิลด์ → Auto-Approved เมื่อ ≥ 90% และผ่านกฎทุกข้อ
- **หน้าเว็บมือถือ** ถ่ายรูปจากกล้อง เตือนภาพเบลอก่อนส่ง ดูผลพร้อมกรอบไฮไลท์ และตรวจแก้ได้บนมือถือ
- **Streamlit HITL Review UI** สำหรับเจ้าหน้าที่ QA บนคอมพิวเตอร์
- สคริปต์ประเมินผล: **CER, Field-Level Accuracy, First-Pass Yield, Exception Rate, Auto-Approved Precision**

---

## 1. สถาปัตยกรรมระบบ (5-Layer Architecture)

```mermaid
flowchart LR
    A[📱 มือถือ / Streamlit / API client] -->|ภาพ| L1
    subgraph L1[1. Intake Layer]
      P1[Resolution] --> P2[Denoise] --> P3[Deskew] --> P4[CLAHE Contrast] --> P5[De-blur]
    end
    L1 --> L2
    subgraph L2[2. Capture & Recognition]
      E1[OCR Engine x3 passes<br/>TrOCR / VLM / Mock] --> K[KIE: EXP / MFG / LOT] --> C[Consensus vote]
    end
    L2 --> L3
    subgraph L3[3. Confidence & Business Rules]
      F[Field confidence] --> R[Rules R01-R09] --> RT{conf ≥ 90%<br/>และผ่านทุกกฎ?}
    end
    RT -->|ใช่| AA[AUTO_APPROVED]
    RT -->|ไม่| Q
    subgraph L4[4. HITL Review Queue]
      Q[คิวตรวจ] --> H[Confirm / Edit / Reject] --> AU[(Audit Trail)]
    end
    AA --> L5
    H --> L5
    subgraph L5[5. Downstream & Feedback Loop]
      J[JSON export → ERP/WMS]
      FB[(Feedback dataset<br/>ภาพ crop + ข้อความถูก)] --> FT[Fine-tune TrOCR] -.-> E1
    end
```

| ชั้น | โมดูล | หน้าที่ |
|---|---|---|
| 1. Intake | `app/intake/preprocessing.py`, `augmentation.py` | ปรับความละเอียด, ลด noise, แก้ภาพเอียง, เพิ่มคอนทราสต์ (CLAHE), แก้ภาพเบลอ (Unsharp / Wiener) · Data augmentation |
| 2. Recognition | `app/recognition/*`, `app/extraction/*` | OCR/VLM อ่านข้อความแบบ fine-grained พร้อมความมั่นใจรายตัวอักษร · KIE สกัด EXP/MFG/LOT · Ensemble |
| 3. Confidence & Rules | `app/rules/*` | คะแนนความมั่นใจระดับฟิลด์ · กฎธุรกิจ · Routing |
| 4. HITL Review | `app/review/service.py`, `ui/review_app.py`, `web/index.html` | คิวตรวจ, ยืนยัน/แก้ไข/ปฏิเสธ, Audit Trail |
| 5. Downstream & Feedback | `app/downstream/*`, `scripts/finetune_trocr.py` | ส่งออก JSON · Feedback dataset · Fine-tune โมเดล |

## 2. โครงสร้างโปรเจค

```
expiry-ocr/
├── app/
│   ├── main.py                 # FastAPI entrypoint (+ เสิร์ฟหน้าเว็บมือถือที่ /)
│   ├── config.py               # ค่าคอนฟิกจาก .env
│   ├── db.py, models.py        # SQLAlchemy: SQLite (ค่าเริ่มต้น) / PostgreSQL
│   ├── schemas.py              # Pydantic request bodies
│   ├── pipeline.py             # Orchestrator ของทั้ง 5 ชั้น (ไม่ผูก DB)
│   ├── api/routes.py           # REST API v1
│   ├── intake/                 # [ชั้น 1] preprocessing.py, augmentation.py
│   ├── recognition/            # [ชั้น 2] base.py, trocr_engine.py, vlm_engine.py, mock_engine.py, detector.py
│   ├── extraction/             # [ชั้น 2] date_parser.py, kie.py, consensus.py
│   ├── rules/                  # [ชั้น 3] confidence.py, validators.py, router.py
│   ├── review/service.py       # [ชั้น 4] HITL queue, audit trail, feedback
│   └── downstream/             # [ชั้น 5] exporter.py, visualize.py
├── web/index.html              # หน้าเว็บมือถือ (ถ่ายรูป / ดูผล / ตรวจแก้)
├── ui/review_app.py            # Streamlit HITL Review UI
├── evaluation/                 # metrics.py, evaluate.py
├── scripts/                    # generate_synthetic.py, finetune_trocr.py
├── tests/                      # pytest 33 ข้อ
├── requirements.txt            # แกนหลัก (ไม่ต้องมี GPU)
└── requirements-ml.txt         # torch + transformers สำหรับ TrOCR
```

## 3. ติดตั้งและรัน

### วิธีง่ายสุด (Windows): ดับเบิลคลิก `run.bat`
ครั้งแรกจะสร้าง `.venv` และติดตั้งแพ็กเกจให้เอง (ใช้เวลาสักครู่) จากนั้นเปิดเซิร์ฟเวอร์และเปิดเบราว์เซอร์ไปที่
`http://localhost:8000/` อัตโนมัติ หน้าต่างดำจะแสดงลิงก์สำหรับเปิดจากมือถือด้วย (ต้องเปิดหน้าต่างนี้ค้างไว้ตลอด)
- macOS / Linux: `./run.sh`
- VS Code: กด **F5** แล้วเลือก "Run ทั้งระบบ"
- ถ้าเว็บขึ้น `ERR_CONNECTION_REFUSED` แปลว่าเซิร์ฟเวอร์ยังไม่ได้เปิดหรือถูกปิดไปแล้ว ให้รัน `run.bat` ใหม่

### ติดตั้งเอง

ต้องใช้ Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # ปรับค่าตามต้องการ

# 1) สร้างภาพตัวอย่าง (ภาพสังเคราะห์ + ground truth) สำหรับสาธิต
python -m scripts.generate_synthetic --out data/synthetic --n 300 --seed 7

# 2) เปิด API (+ หน้าเว็บมือถือ)
uvicorn app.main:app --host 0.0.0.0 --port 8000
#    Swagger:        http://localhost:8000/docs
#    หน้าเว็บมือถือ:  http://localhost:8000/

# 3) เปิดหน้าจอ HITL สำหรับ QA (อีก terminal)
streamlit run ui/review_app.py

# 4) ประเมินผล
python -m evaluation.evaluate --data data/synthetic --out reports
# 5) ทดสอบ
pytest -q
```

### ใช้งานจากมือถือ
1. ให้คอมพิวเตอร์ที่รัน API และมือถืออยู่ Wi-Fi เดียวกัน แล้วเปิด `http://<IP คอมพิวเตอร์>:8000/` บนมือถือ
2. ปุ่ม **"เลือกรูป / ถ่ายด้วยแอปกล้อง"** ใช้ได้ทันทีผ่าน HTTP (เปิดแอปกล้องของมือถือ)
3. ปุ่ม **"เปิดกล้องถ่ายวันหมดอายุ"** (กล้องสดพร้อมกรอบเล็ง) เบราว์เซอร์อนุญาตเฉพาะ **HTTPS** — ใช้
   `cloudflared tunnel --url http://localhost:8000` หรือ `ngrok http 8000` แล้วเปิดลิงก์ https ที่ได้
   หรือรัน uvicorn ด้วย `--ssl-keyfile/--ssl-certfile`
4. หน้าเว็บตรวจความคมชัดของภาพก่อนส่ง ถ้าเบลอจะแนะนำให้ถ่ายใหม่ (ลดความผิดพลาดตั้งแต่ต้นทาง)
5. ถ้าผลเป็น "รอคนตรวจ" สามารถแก้ค่าและกดอนุมัติบนมือถือได้ทันที (บันทึก Audit + Feedback เหมือน Streamlit)

> ค่าเริ่มต้น `OCR_ENGINE=rapidocr` อ่านภาพถ่ายจริงได้ทันที (engine `mock` ใช้สำหรับเทสต์อัตโนมัติเท่านั้น)

## 4. OCR Engines

**RapidOCR** (`OCR_ENGINE=rapidocr`, ค่าเริ่มต้น) — PP-OCR (DBNet text detection + SVTR recognition) แบบ ONNX
ติดมากับ `requirements.txt` รันบน CPU ~0.5 วินาที/รอบ ระบบรวมคำที่อ่านได้เป็นบรรทัด แล้วส่งต่อ KIE


**TrOCR** (`OCR_ENGINE=trocr`)
```bash
pip install -r requirements-ml.txt
# .env
OCR_ENGINE=trocr
TROCR_MODEL=microsoft/trocr-base-printed     # หรือ path ของโมเดลที่ fine-tune แล้ว
```
ระบบหา "บรรทัดข้อความ" ด้วย OpenCV แล้วส่งทีละบรรทัดเข้า TrOCR และคำนวณความมั่นใจรายตัวอักษรจาก
probability ของ token (`compute_transition_scores`) — รันบน CPU ได้แต่ช้า (แนะนำ GPU)

**Vision-Language Model** (`OCR_ENGINE=vlm`) — ผ่าน OpenAI-compatible endpoint เช่น vLLM
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
# .env
OCR_ENGINE=vlm
VLM_ENDPOINT=http://localhost:8001/v1/chat/completions
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```
ถ้า server ส่ง `logprobs` กลับมา จะใช้คำนวณความมั่นใจรายตัวอักษร ถ้าไม่ส่ง ความมั่นใจจะเป็น 0.85 (ต่ำกว่าเกณฑ์) → เข้าคิวตรวจเสมอ

เพิ่ม engine ใหม่ได้โดยสืบทอด `OCREngine` ใน `app/recognition/base.py` แล้วลงทะเบียนใน `factory.py`

## 5. Confidence & Business Rules

**ความมั่นใจระดับฟิลด์** (`app/rules/confidence.py`)
```
confidence = min(ความมั่นใจรายตัวอักษรในช่วงค่า)
             × 0.93^(จำนวนตัวอักษรที่ต้องแก้ เช่น O→0, I→1)
             × 0.88 (ถ้าไม่มี label EXP/MFG กำกับ ต้องเดาจากลำดับ)
             × agreement (สัดส่วนรอบ ensemble ที่อ่านได้ตรงกัน)
```
ใช้ `min` ไม่ใช่ `mean` เพราะวันที่ผิดแค่ 1 ตัวอักษรก็ผิดทั้งฟิลด์

**กฎธุรกิจ** (`app/rules/validators.py`)

| รหัส | กฎ | ระดับ |
|---|---|---|
| R01 | ต้องพบวันหมดอายุ | error |
| R02 | รูปแบบวันที่อยู่ในรายการมาตรฐาน (YYYY-MM-DD, DD/MM/YYYY, DD/MM/YY, DD MON YYYY, MON YYYY, MM/YYYY, DDMMYY) | error |
| R03 | เดือน 01-12, วัน 01-31, ปีอยู่ในช่วงที่เป็นไปได้ | error |
| R04 | เป็นวันที่ที่มีจริงในปฏิทิน (เช่น 30/02 ไม่ผ่าน) | error |
| R05 | วันหมดอายุต้องอยู่หลังวันผลิต | error |
| R06 | วันผลิตต้องไม่อยู่ในอนาคต | error |
| R07 | อายุสินค้าไม่เกิน `MAX_SHELF_LIFE_YEARS` (ตรวจปีที่อ่านผิด) | error |
| R08 | Lot เป็นตัวอักษร/ตัวเลข 3-20 ตัว | error |
| R09 | สินค้าหมดอายุแล้ว | warning (แจ้งเตือน) |

รองรับปี พ.ศ. (2568 / 68), ชื่อเดือนไทย (ม.ค. – ธ.ค.) และอังกฤษ, label ภาษาไทย (หมดอายุ, ควรบริโภคก่อน, ผลิต, ล็อต)

**Routing** (`app/rules/router.py`): ทุกฟิลด์ความมั่นใจ ≥ `CONFIDENCE_THRESHOLD` (0.90) **และ** ผ่านกฎ error ทุกข้อ **และ** ensemble เห็นตรงกันทุกรอบ → `AUTO_APPROVED`; ไม่เช่นนั้น → `PENDING_REVIEW` พร้อมเหตุผล

## 6. เรื่องความแม่นยำ 99–100%

ไม่มี OCR ตัวไหนรับประกันได้ว่าอ่านถูก 100% ทุกภาพ เพราะภาพจริงมีทั้งเบลอ สะท้อนแสง หมึกจาง และตัวเลขที่ถูกบัง
ระบบนี้จึงออกแบบให้ **"ผลลัพธ์ที่ส่งออกไปถูกต้อง ≥99%"** ด้วยหลักการ:

1. **ป้องกันตั้งแต่ต้นทาง** — หน้าเว็บมือถือเตือนภาพเบลอ + กรอบเล็ง, preprocessing แก้ภาพเอียง/เบลอ/คอนทราสต์ต่ำ
2. **อ่านหลายรอบแล้วโหวต** (ensemble) — ความผิดพลาดแบบ "มั่นใจแต่ผิด" ของรอบใดรอบหนึ่งจะถูกจับได้
3. **ความมั่นใจแบบเข้มงวด + กฎธุรกิจ** — ระบบอนุมัติเองเฉพาะเมื่อมั่นใจจริง ที่เหลือส่งคนตรวจ
4. **มนุษย์ตรวจเคสที่ไม่แน่ใจ** — ผลลัพธ์สุดท้าย = ระบบอัตโนมัติที่แม่นยำสูง + คนตรวจส่วนที่เหลือ
5. **เรียนรู้จากการแก้ไข** — ทุกการแก้ไขกลายเป็นข้อมูล fine-tune ทำให้ FPY สูงขึ้นเรื่อย ๆ

ตัวชี้วัดที่ต้องดูคู่กันคือ **Auto-Approved Precision** (ระบบอนุมัติเองแล้วถูกกี่ %) กับ **First-Pass Yield** (ผ่านเองได้กี่ %)
ปรับ `CONFIDENCE_THRESHOLD` จากตาราง threshold sweep ในรายงานประเมินผล: threshold สูง = แม่นขึ้นแต่คนต้องตรวจมากขึ้น

## 7. ตัวชี้วัดการประเมินผล

```bash
python -m evaluation.evaluate --data data/synthetic --out reports [--engine trocr] [--passes 3] [--threshold 0.9]
```
ได้ `reports/report.md`, `report.json`, `errors.json`

| ตัวชี้วัด | สูตร |
|---|---|
| Character Error Rate (CER) | (Substitution + Deletion + Insertion) / จำนวนตัวอักษรจริง — Levenshtein |
| Field-Level Accuracy | ฟิลด์ที่ค่าตรงกับ ground truth ทั้งหมด / จำนวนฟิลด์ |
| First-Pass Yield | ภาพ AUTO_APPROVED / ภาพทั้งหมด |
| Exception Rate | ภาพที่เข้า HITL Review Queue / ภาพทั้งหมด |
| Auto-Approved Precision | ภาพ AUTO_APPROVED ที่ถูกทุกฟิลด์ / ภาพ AUTO_APPROVED |
| Final Accuracy (with HITL) | ความถูกต้องของผลลัพธ์สุดท้าย (สมมติคนตรวจแก้ถูก) |

API `GET /api/v1/metrics` ให้ First-Pass Yield, Exception Rate และอัตราการแก้ไขจากการใช้งานจริงด้วย

**ชุดข้อมูลจริงสำหรับรายงาน**: ถ่ายฉลากจริง 300–1,000 ภาพ ใส่ ground truth ในไฟล์ `manifest.jsonl` รูปแบบเดียวกับ
`data/synthetic/manifest.jsonl` แล้วรัน evaluate ด้วย `--engine trocr` หรือ `vlm`

### ผลตัวอย่าง (Mock engine บนภาพสังเคราะห์ 300 ภาพ, ensemble 3 รอบ, threshold 0.90)

| ตัวชี้วัด | ผล |
|---|---|
| CER (EXP) | 0.83% |
| Field-Level Accuracy (EXP / MFG / LOT) | 99.00% / 99.00% / 96.67% |
| First-Pass Yield | 70.67% |
| Exception Rate | 29.33% |
| **Auto-Approved Precision** | **100% (212/212, False Accept = 0)** |
| Final Accuracy (with HITL) | 100% |

ภาพสังเคราะห์ผ่าน augmentation หนัก (เอียง ±10°, เบลอ, หมึกบาง/หนา, คอนทราสต์ต่ำ, noise) และ mock engine
จำลองความผิดพลาดแบบ OCR จริง รวมถึง "ผิดแต่มั่นใจ" — ตัวเลขนี้ **ใช้แสดงว่ากลไก routing ทำงาน** เท่านั้น
ผลที่จะใส่ในเล่มโปรเจคต้องรันใหม่ด้วย TrOCR/VLM บนภาพถ่ายจริง

จากตาราง threshold sweep: threshold 0.50 ทำให้ FPY 86% แต่มีภาพอนุมัติผิด 3 ภาพ ส่วน 0.70 ขึ้นไปไม่มีภาพผิดเลย
แสดงให้เห็น trade-off ระหว่างความแม่นยำกับภาระงานของคน

## 8. REST API

| Method | Path | คำอธิบาย |
|---|---|---|
| POST | `/api/v1/documents` | อัปโหลดภาพ (multipart `file`) → ประมวลผล → คืน JSON พร้อมสถานะ |
| GET | `/api/v1/documents?status=` | รายการเอกสาร |
| GET | `/api/v1/documents/{id}` | รายละเอียด + audit trail |
| GET | `/api/v1/documents/{id}/image?variant=annotated\|original\|processed` | ภาพพร้อมกรอบไฮไลท์ |
| GET | `/api/v1/documents/{id}/audit` | Audit Trail |
| GET | `/api/v1/review/queue?order=fifo\|hardest` | คิว HITL |
| POST | `/api/v1/review/{id}/submit` | ยืนยัน/แก้ไข `{"reviewer","corrections":[{"name","value","raw_text","reason"}],"force"}` |
| POST | `/api/v1/review/{id}/reject` | ปฏิเสธ `{"reviewer","reason"}` |
| GET | `/api/v1/export/documents` | JSON ผลลัพธ์สุดท้ายส่ง downstream |
| GET | `/api/v1/export/feedback` | zip ของ feedback dataset (`feedback.jsonl` + crops) |
| GET | `/api/v1/metrics` | FPY, Exception Rate, Correction Rate |

ตัวอย่างผลลัพธ์ (ย่อ):
```json
{
  "document_id": 12, "status": "AUTO_APPROVED", "overall_confidence": 0.9931,
  "fields": {
    "exp_date":   {"value": "2027-06-01", "predicted_raw": "01/06/2027", "confidence": 0.9931, "date_format": "DD/MM/YYYY", "bbox": [[52,88],[402,88],[402,130],[52,130]]},
    "mfg_date":   {"value": "2026-06-01", "confidence": 0.9952},
    "lot_number": {"value": "A2506B", "confidence": 0.9964}
  },
  "audit_trail": [{"who": "system", "action": "AUTO_APPROVE", "when": "..."}]
}
```

## 9. ฐานข้อมูล

| ตาราง | ข้อมูลหลัก |
|---|---|
| `documents` | ไฟล์, sha256, สถานะ, ความมั่นใจรวม, เหตุผล routing, ผลกฎ, ข้อมูล preprocessing, ข้อความ OCR |
| `extracted_fields` | ชื่อฟิลด์, ค่าดิบ, ค่าทำนาย, ค่าสุดท้าย, ความมั่นใจ, รูปแบบวันที่, กรอบ (polygon), แก้ไขหรือไม่ |
| `audit_logs` | **ใคร (actor), เมื่อไร, การกระทำ, ฟิลด์, ค่าเดิม, ค่าใหม่, เหตุผล** |
| `feedback_samples` | ภาพต้นฉบับ, ภาพ crop, ข้อความที่ทำนาย, ข้อความที่ถูกต้อง, OCR ผิดหรือไม่, ใช้เทรนแล้วหรือยัง |

เปลี่ยนเป็น PostgreSQL: `pip install "psycopg[binary]"` แล้วตั้ง
`DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/expiry_ocr` (ตารางสร้างอัตโนมัติตอนเริ่ม API)

## 10. Feedback Loop (ปรับปรุงโมเดล)

1. เจ้าหน้าที่ตรวจ/แก้ในคิว → ระบบเก็บ crop ของบรรทัด + ข้อความที่ถูกต้องใน `feedback_samples`
2. ดาวน์โหลด `GET /api/v1/export/feedback` (หรือแท็บ Feedback Data ใน Streamlit) แตกไฟล์ไว้ที่ `data/feedback`
3. `python -m scripts.finetune_trocr --data data/feedback --out models/trocr-expiry --epochs 10`
4. ตั้ง `TROCR_MODEL=models/trocr-expiry` แล้ววัดผลเทียบก่อน-หลังด้วย `evaluation.evaluate`

## 11. ข้อจำกัดและงานต่อยอด
- Text detector เป็นแบบ classical (OpenCV) — ต่อยอดเป็น DBNet/CRAFT สำหรับฉลากโค้ง/พื้นหลังซับซ้อน
- Mock engine ใช้สาธิตและทดสอบระบบเท่านั้น ตัวเลขประเมินผลในรายงานจริงต้องมาจาก TrOCR/VLM บนภาพถ่ายจริง
- ต่อยอด ensemble แบบหลาย engine (TrOCR + VLM ต้องตรงกันจึงอนุมัติ) เพื่อลด false accept บนภาพจริง
- เพิ่มระบบผู้ใช้/สิทธิ์ (authentication) ก่อนใช้งานจริง

## 12. เผยแพร่ให้คนอื่นเปิดหน้าเว็บได้ (ฟรี)

GitHub Pages เปิดได้เฉพาะหน้าเว็บนิ่ง แต่ระบบนี้ต้องมีเซิร์ฟเวอร์ Python สำหรับ AI OCR จึงใช้ **Hugging Face Spaces** (ฟรี, CPU 16GB)

**ครั้งแรก (ประมาณ 5 นาที)**
1. สมัคร/ล็อกอิน https://huggingface.co → **New Space** → ตั้งชื่อ (เช่น `expiry-ocr`) → SDK เลือก **Docker** (Blank) → Public → Create
2. สร้าง token ที่ https://huggingface.co/settings/tokens (สิทธิ์ **Write**)
3. ใน GitHub repo: **Settings → Secrets and variables → Actions**
   - แท็บ Secrets: เพิ่ม `HF_TOKEN` = token จากข้อ 2
   - แท็บ Variables: เพิ่ม `HF_SPACE` = `<ชื่อผู้ใช้ HF>/<ชื่อ Space>` เช่น `jirayu/expiry-ocr`
4. ไปที่แท็บ **Actions → Deploy to Hugging Face Spaces → Run workflow**
5. รอ Space build เสร็จ (~5 นาที) จะได้ลิงก์ `https://<ชื่อผู้ใช้>-expiry-ocr.hf.space` ส่งให้ใครก็เปิดได้
   และเป็น HTTPS จึงใช้ปุ่ม "เปิดกล้อง" บนมือถือได้

หลังจากนั้นทุกครั้งที่ push เข้า `main` ระบบจะ deploy ใหม่ให้อัตโนมัติ

ทางเลือกอื่น: `Dockerfile` ใช้กับ Render / Railway / Google Cloud Run ได้เช่นกัน (ตั้ง port ผ่าน env `PORT`)

> หมายเหตุ: บน Spaces ฟรี ฐานข้อมูลและภาพที่อัปโหลดจะหายเมื่อ Space รีสตาร์ท (เหมาะกับการสาธิต)
> และทุกคนที่มีลิงก์สามารถตรวจ/แก้ในคิวได้ ถ้าใช้งานจริงควรเพิ่มระบบล็อกอิน
