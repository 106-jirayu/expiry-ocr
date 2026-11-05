# ชุดทดสอบภาพฉลากจริง (Real-label test set) — 36 ภาพ

ภาพถ่ายบรรจุภัณฑ์จริงจากร้านค้า (ถ่ายด้วยมือถือ ถือในมือ แสงร้าน มีเงา/ฟิล์มสะท้อน) พร้อมเฉลยวันหมดอายุที่ **อ่านและตรวจด้วยตาทีละภาพ**
ใช้วัดผล AI OCR บนภาพจริง แยกจากชุด synthetic

## ที่มาและลิขสิทธิ์

ทุกภาพมาจาก **ExpDate dataset** ซึ่งเผยแพร่ภายใต้ **CC BY 4.0**
(ใช้ แชร์ ดัดแปลงได้ทั้งงานวิจัยและเชิงพาณิชย์ โดยต้องให้เครดิต):

> A. C. Seker and S. C. Ahn, "A generalized framework for recognition of expiration dates on product packages
> using fully convolutional networks," *Expert Systems with Applications*, vol. 203, 117310, 2022.
> https://doi.org/10.1016/j.eswa.2022.117310 · โครงการ: https://felizang.github.io/expdate/

- `real_001`–`real_032`: ภาพชุด Products ของ ExpDate ที่มีผู้นำมาเผยแพร่ซ้ำใน
  https://github.com/MonaKhaled10/Expiration-Date-Recognition/tree/main/Test-images (ชื่อไฟล์เดิมอยู่ในคอลัมน์ `source_file`)
- `real_033`–`real_036`: ภาพตัวอย่างจาก repo ทางการ https://github.com/felizang/expdate/tree/main/images (ความละเอียดต่ำ ~500px)

ภาพไม่ได้ถูกแก้ไข เปลี่ยนแค่ชื่อไฟล์

## ไฟล์

| ไฟล์ | เนื้อหา |
|---|---|
| `images/real_XXX.jpg` | ภาพต้นฉบับ |
| `labels.csv` | เฉลยแบบอ่านง่าย: วันหมดอายุ (ISO), ข้อความที่พิมพ์บนฉลาก, MFG, LOT, คำนำหน้า, รูปแบบวันที่, ภาษา, หมายเหตุ, แหล่งที่มา |
| `manifest.jsonl` | รูปแบบเดียวกับ `data/synthetic/manifest.jsonl` ใช้กับ `evaluation.evaluate` ได้ทันที |

## หลักการทำเฉลย

- อ่านเฉพาะวันที่ที่ชัดเจน ภาพที่กำกวม **ตัดทิ้ง** (6 ภาพ จากทั้งหมด 44 ภาพที่ดาวน์โหลดได้; อีก 2 ภาพตัวอย่างความละเอียดต่ำไม่ได้ใช้): วันที่ไม่มีปี (`02.18까지`, `01.24`),
  วันที่ที่ไม่ใช่วันหมดอายุ (สินค้าไม่ระบุวันหมดอายุ), มีสองวันโดยไม่มีคำกำกับ, `03/04/23` ที่แยก DD/MM กับ MM/DD ไม่ได้, รหัส `E 28 10 21` ที่ไม่แน่ใจความหมาย
- ฉลากที่ระบุแค่เดือน/ปี (`BEST BY JUN 2022`) → วันสุดท้ายของเดือน (ตรงกับ convention ของ `date_parser`)
- ฉลากเกาหลี `유통기한 X 부터 Y 까지` → X = วันผลิต (mfg), Y = วันหมดอายุ (exp)
- LOT ใส่เฉพาะเมื่อมีคำว่า LOT กำกับชัด (2 ภาพ) นอกนั้นเว้นว่าง
- 7 ภาพมีแค่วันที่ไม่มีคำกำกับ (`label_keyword = (none)`) — เป็นเคสจริงที่พบบ่อย เฉลยถือว่าวันนั้นเป็นวันหมดอายุ

## องค์ประกอบของชุด

- ภาษา: อังกฤษ 24 ภาพ (บางภาพมีอาหรับ/ฝรั่งเศส/ญี่ปุ่น/เวียดนามร่วม), เกาหลีล้วน 12 ภาพ, สินค้าผลิตในไทย 1 ภาพ
- รูปแบบวันที่: `YYYY.MM.DD`, `YY.MM.DD`, `DD/MM/YYYY`, `DD.MM.YYYY`, `MM/DD/YY`, `MON DD YYYY`, `DDMONYY`, `MONDDYYYY`, `MON YYYY` ฯลฯ
- **ข้อจำกัด: ไม่มีฉลากภาษาไทย/ปี พ.ศ.** — ไม่พบชุดข้อมูลภาพฉลากไทยที่เปิดลิขสิทธิ์ ควรเพิ่มภาพถ่ายฉลากไทยเองในรูปแบบ `labels.csv` เดียวกัน

## ผลการประเมิน (RapidOCR, ensemble 3 รอบ, threshold 0.90)

```bash
python -m evaluation.evaluate --data datasets/real_expdate --out reports/real_expdate --engine rapidocr
```

รายงานเต็ม: [`reports/real_expdate/report.md`](../../reports/real_expdate/report.md), รายภาพที่ผิด: `reports/real_expdate/errors.json`

| ตัวชี้วัด | ค่า |
|---|---|
| Field-Level Accuracy (EXP) | **52.78%** (19/36) |
| CER (EXP) | 36.47% |
| First-Pass Yield (อนุมัติอัตโนมัติ) | 16.67% (6/36) |
| Auto-Approved Precision (ทุกฟิลด์) | 83.33% (5/6) — EXP ถูกครบ 6/6, เคสที่ผิด 1 เคสคืออ่าน MFG ไม่เจอ |
| Final Accuracy (with HITL) | 97.22% |

แยกย่อย EXP accuracy: มีคำกำกับ 16/29 · ไม่มีคำกำกับ 3/7 · เกาหลีล้วน 7/12 · ละติน 12/24 · ภาพความละเอียดต่ำ 2/4
ใน 17 ภาพที่ผิด: อ่านไม่เจอเลย 11 ภาพ, อ่านได้แต่ค่าผิด 6 ภาพ (ทั้งหมดถูกส่งเข้า HITL queue ไม่มีภาพใดที่วันหมดอายุผิดแล้วผ่านอัตโนมัติ)
