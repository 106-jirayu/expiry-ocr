"""สคริปต์ประเมินผลโปรเจค

ใช้งาน:
  python -m scripts.generate_synthetic --out data/synthetic --n 300 --seed 7
  python -m evaluation.evaluate --data data/synthetic --out reports/ [--engine auto|rapidocr|trocr|vlm|mock] [--passes 3]

Confidence calibration (ทำครั้งเดียวต่อ engine บนชุดที่มีเฉลย แล้วประเมินบน "อีกชุด" ที่ไม่ได้ใช้ fit):
  python -m evaluation.evaluate --data data/calib --engine rapidocr --fit-calibration
  python -m evaluation.evaluate --data data/test  --engine rapidocr

ชุดข้อมูลจริง: เตรียม manifest.jsonl ในรูปแบบเดียวกัน
  {"image": "images/x.jpg", "ground_truth": {"exp_date": "2026-12-31", "mfg_date": null, "lot_number": "A123",
   "exp_text": "31/12/2026", "mfg_text": null, "lot_text": "A123"}}
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.config import settings
from app.extraction.kie import FIELDS
from app.pipeline import ExpiryOCRPipeline
from app.recognition.factory import build_engine
from app.rules.calibration import Calibrator, fit
from app.rules.router import route
from evaluation.metrics import cer, field_accuracy


def ratio(n: int, d: int) -> float:
    return n / d if d else 0.0

TEXT_KEY = {"exp_date": "exp_text", "mfg_date": "mfg_text", "lot_number": "lot_text"}


def run(data_dir: Path, engine: str | None, passes: int | None, threshold: float, today: date | None,
        calibrator: Calibrator | None = None):
    rows = [json.loads(line) for line in open(data_dir / "manifest.jsonl", encoding="utf-8") if line.strip()]
    pipe = ExpiryOCRPipeline(engine=build_engine(engine), threshold=threshold, passes=passes, today=today,
                             calibrator=calibrator)
    records = []
    for i, r in enumerate(rows, 1):
        res = pipe.run((data_dir / r["image"]).read_bytes())
        records.append({"row": r, "result": res})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}")
    return records, pipe


def correct_all(res, gt) -> bool:
    return all((res.fields[n].value if n in res.fields else None) == gt.get(n) for n in FIELDS)


def summarize(records, threshold: float) -> dict:
    n = len(records)
    out: dict = {"images": n, "threshold": threshold, "fields": {}}
    for name in FIELDS:
        preds, gts, raw_p, raw_r = [], [], [], []
        for rec in records:
            gt, f = rec["row"]["ground_truth"], rec["result"].fields.get(name)
            preds.append(f.value if f else None)
            gts.append(gt.get(name))
            if gt.get(TEXT_KEY[name]):
                raw_p.append(f.raw_text if f else None)
                raw_r.append(gt[TEXT_KEY[name]])
        present = [(p, g) for p, g in zip(preds, gts) if g is not None]
        out["fields"][name] = {
            "accuracy_all_images": round(field_accuracy(preds, gts), 4),
            "accuracy_when_present": round(field_accuracy([p for p, _ in present], [g for _, g in present]), 4)
            if present else None,
            "cer": round(cer(raw_p, raw_r), 4) if raw_r else None,
            "support": len(present),
        }
    auto = [r for r in records if r["result"].decision.auto_approved]
    auto_ok = sum(correct_all(r["result"], r["row"]["ground_truth"]) for r in auto)
    raw_ok = sum(correct_all(r["result"], r["row"]["ground_truth"]) for r in records)
    exceptions = n - len(auto)
    out.update({
        "cer_exp": out["fields"]["exp_date"]["cer"],
        "field_accuracy_exp": out["fields"]["exp_date"]["accuracy_all_images"],
        "document_accuracy_raw_ocr": round(ratio(raw_ok, n), 4),
        "first_pass_yield": round(ratio(len(auto), n), 4),
        "exception_rate": round(ratio(exceptions, n), 4),
        "auto_approved": len(auto),
        "auto_approved_correct": auto_ok,
        "auto_approved_precision": round(ratio(auto_ok, len(auto)), 4) if auto else None,
        "false_accepts": len(auto) - auto_ok,
        # สมมติผู้ตรวจแก้เคสที่เข้าคิวได้ถูกต้อง -> ความถูกต้องของผลลัพธ์สุดท้ายที่ส่ง downstream
        "final_accuracy_with_hitl": round(ratio(auto_ok + exceptions, n), 4),
        "caught_errors_by_hitl": sum(not correct_all(r["result"], r["row"]["ground_truth"])
                                     for r in records if not r["result"].decision.auto_approved),
    })
    return out


def threshold_sweep(records, thresholds) -> list[dict]:
    """ผลของการปรับ threshold ต่อ FPY vs ความแม่นยำของ Auto-Approved (ใช้เลือก threshold ในรายงาน)"""
    rows = []
    for th in thresholds:
        auto_n = auto_ok = 0
        for rec in records:
            res = rec["result"]
            if route(res.fields, res.rules, th).auto_approved:
                auto_n += 1
                auto_ok += correct_all(res, rec["row"]["ground_truth"])
        n = len(records)
        rows.append({"threshold": th, "first_pass_yield": round(auto_n / n, 4),
                     "exception_rate": round(1 - auto_n / n, 4),
                     "auto_precision": round(auto_ok / auto_n, 4) if auto_n else None,
                     "false_accepts": auto_n - auto_ok})
    return rows


def to_markdown(s: dict, sweep: list[dict], meta: dict) -> str:
    pct = lambda v: "-" if v is None else f"{v * 100:.2f}%"  # noqa: E731
    lines = [
        "# รายงานผลการประเมิน Expiry Date AI OCR", "",
        f"- ชุดข้อมูล: `{meta['data']}` ({s['images']} ภาพ)",
        f"- OCR engine: `{meta['engine']}` · ensemble passes: {meta['passes']} · threshold: {s['threshold']:.0%}", "",
        "## ตัวชี้วัดหลัก", "",
        "| ตัวชี้วัด | ค่า | ความหมาย |", "|---|---|---|",
        f"| Character Error Rate (EXP) | {pct(s['cer_exp'])} | ความผิดพลาดระดับตัวอักษรของข้อความวันหมดอายุ (ยิ่งต่ำยิ่งดี) |",
        f"| Field-Level Accuracy (EXP) | {pct(s['field_accuracy_exp'])} | วันหมดอายุถูกต้องตรงทั้งฟิลด์ (ก่อนคนตรวจ) |",
        f"| Document Accuracy (OCR ล้วน) | {pct(s['document_accuracy_raw_ocr'])} | ทุกฟิลด์ในภาพถูกหมด ก่อนคนตรวจ |",
        f"| First-Pass Yield | {pct(s['first_pass_yield'])} | ผ่านอัตโนมัติโดยไม่ต้องใช้คน |",
        f"| Exception Rate | {pct(s['exception_rate'])} | ถูกส่งเข้า HITL Review Queue |",
        f"| **Auto-Approved Precision** | **{pct(s['auto_approved_precision'])}** | ภาพที่ระบบอนุมัติเองแล้วถูกจริง ({s['auto_approved_correct']}/{s['auto_approved']}) |",
        f"| False Accepts | {s['false_accepts']} | ภาพที่ระบบอนุมัติเองแต่ผิด (ต้องใกล้ 0) |",
        f"| **Final Accuracy (with HITL)** | **{pct(s['final_accuracy_with_hitl'])}** | ผลลัพธ์สุดท้ายที่ส่ง downstream (สมมติคนตรวจแก้ถูก) |",
        "", "## รายฟิลด์", "", "| ฟิลด์ | Accuracy (ทุกภาพ) | Accuracy (เมื่อมีบนฉลาก) | CER | จำนวน |", "|---|---|---|---|---|",
    ]
    for name, f in s["fields"].items():
        lines.append(f"| {name} | {pct(f['accuracy_all_images'])} | {pct(f['accuracy_when_present'])} | "
                     f"{pct(f['cer'])} | {f['support']} |")
    lines += ["", "## Threshold sweep (FPY vs ความแม่นยำ)", "",
              "| Threshold | First-Pass Yield | Exception Rate | Auto Precision | False Accepts |", "|---|---|---|---|---|"]
    for r in sweep:
        lines.append(f"| {r['threshold']:.2f} | {pct(r['first_pass_yield'])} | {pct(r['exception_rate'])} | "
                     f"{pct(r['auto_precision'])} | {r['false_accepts']} |")
    return "\n".join(lines) + "\n"


def fit_and_save(records, threshold: float, data: Path) -> None:
    raw, ok = [], []
    for rec in records:
        gt = rec["row"]["ground_truth"]
        for name, f in rec["result"].fields.items():
            raw.append(f.confidence)
            ok.append(f.value is not None and f.value == gt.get(name))
    engine = records[0]["result"].engine if records else "unknown"
    points = fit(raw, ok)
    cal = Calibrator(settings.calibration_file)
    cal.save(engine, points, {"fitted_on": str(data), "fields": len(raw), "field_accuracy": round(sum(ok) / len(ok), 4)})
    print(f"บันทึก calibration ของ '{engine}' ({len(raw)} ฟิลด์) ที่ {settings.calibration_file}: {points}")
    for rec in records:  # ใช้คะแนนที่ calibrate แล้วกับรายงานรอบนี้
        res = rec["result"]
        cal.apply(res.fields, res.engine)
        res.decision = route(res.fields, res.rules, threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--today", default=None)
    ap.add_argument("--fit-calibration", action="store_true",
                    help="fit ตาราง calibration ของ engine นี้จากชุดข้อมูลนี้ แล้วบันทึกลง CALIBRATION_FILE")
    a = ap.parse_args()
    today = date.fromisoformat(a.today) if a.today else None
    records, pipe = run(Path(a.data), a.engine, a.passes, a.threshold, today,
                        Calibrator("__none__") if a.fit_calibration else None)
    if a.fit_calibration:
        fit_and_save(records, a.threshold, Path(a.data))
    summary = summarize(records, a.threshold)
    sweep = threshold_sweep(records, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"data": a.data, "engine": pipe.engine.name, "passes": pipe.passes}
    (out / "report.json").write_text(json.dumps({"meta": meta, "summary": summary, "threshold_sweep": sweep},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    errors = [{"image": r["row"]["image"], "status": r["result"].decision.status.value,
               "gt": r["row"]["ground_truth"], "pred": {k: v.value for k, v in r["result"].fields.items()},
               "reasons": r["result"].decision.reasons}
              for r in records if not correct_all(r["result"], r["row"]["ground_truth"])]
    (out / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(summary, sweep, meta)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
