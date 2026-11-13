import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from scripts.generate_synthetic import generate


def _images(tmp_path, n=12):
    generate(tmp_path, n, seed=3)
    rows = [json.loads(x) for x in open(tmp_path / "manifest.jsonl")]
    return [(tmp_path / r["image"], r["ground_truth"]) for r in rows]


def test_end_to_end_review_flow(tmp_path: Path):
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        before = c.get("/api/v1/metrics").json()["total_documents"]
        docs = []
        for path, gt in _images(tmp_path):
            r = c.post("/api/v1/documents", files={"file": (path.name, path.read_bytes(), "image/png")})
            assert r.status_code == 200, r.text
            docs.append((r.json(), gt))
        statuses = {d["status"] for d, _ in docs}
        assert statuses <= {"AUTO_APPROVED", "PENDING_REVIEW"}

        queue = c.get("/api/v1/review/queue").json()
        mine = {d["document_id"] for d, _ in docs if d["status"] == "PENDING_REVIEW"}
        assert mine <= {q["document_id"] for q in queue}
        if not mine:
            return
        doc_id = min(mine)
        gt = next(g for d, g in docs if d["document_id"] == doc_id)

        # ค่าที่ผิดกฎ (EXP ก่อน MFG) ต้องถูกปฏิเสธ เว้นแต่ force
        bad = c.post(f"/api/v1/review/{doc_id}/submit", json={
            "reviewer": "qa1", "corrections": [{"name": "exp_date", "value": "2000-01-01"},
                                               {"name": "mfg_date", "value": "2001-01-01"}]})
        assert bad.status_code == 422

        corr = [{"name": n, "value": gt[n], "reason": "แก้ตามฉลาก"} for n in ("exp_date", "mfg_date", "lot_number")]
        ok = c.post(f"/api/v1/review/{doc_id}/submit", json={"reviewer": "qa1", "corrections": corr})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["status"] in ("HUMAN_APPROVED", "HUMAN_CORRECTED")
        assert body["fields"]["exp_date"]["value"] == gt["exp_date"]
        audit = c.get(f"/api/v1/documents/{doc_id}/audit").json()
        assert any(a["who"] == "qa1" for a in audit)
        assert all({"who", "when", "original_value", "corrected_value", "reason"} <= set(a) for a in audit)

        # ส่งซ้ำไม่ได้ (ไม่อยู่ในคิวแล้ว)
        assert c.post(f"/api/v1/review/{doc_id}/submit", json={"reviewer": "qa1"}).status_code == 409

        assert c.get(f"/api/v1/documents/{doc_id}/image?variant=annotated").headers["content-type"] == "image/png"
        m = c.get("/api/v1/metrics").json()
        assert m["total_documents"] - before == len(docs)
        assert abs(m["first_pass_yield"] + m["exception_rate"] - 1) < 1e-9
        fb = c.get("/api/v1/export/feedback")
        assert fb.status_code == 200 and fb.headers["content-type"] == "application/zip"
        assert isinstance(c.get("/api/v1/export/documents").json(), list)


def test_reject_and_bad_upload():
    with TestClient(app) as c:
        assert c.post("/api/v1/documents", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400
        assert c.post("/api/v1/review/99999/reject", json={"reviewer": "a", "reason": "b"}).status_code == 404


def test_manual_entry():
    from datetime import date, timedelta
    exp = (date.today() + timedelta(days=3)).isoformat()
    with TestClient(app) as c:
        r = c.post("/api/v1/documents/manual", json={"user": "qa1", "exp_date": exp, "lot_number": "a12",
                                                      "product_name": "นมกล่อง"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["status"] == "HUMAN_APPROVED" and d["engine"] == "manual" and d["filename"] == "นมกล่อง"
        assert d["fields"]["exp_date"]["value"] == exp and d["fields"]["lot_number"]["value"] == "A12"
        assert d["expiry"]["status"] == "EXPIRING_SOON"
        assert c.get(f"/api/v1/documents/{d['document_id']}/image").status_code == 404
        cal = c.get(f"/api/v1/calendar?start={exp}&end={exp}").json()
        assert d["document_id"] in {x["document_id"] for x in cal}

        assert c.post("/api/v1/documents/manual", json={"user": "qa1", "exp_date": "31/12/2027"}).status_code == 422
        bad = {"user": "qa1", "exp_date": "2000-01-01", "mfg_date": "2001-01-01"}
        assert c.post("/api/v1/documents/manual", json=bad).status_code == 422
        assert c.post("/api/v1/documents/manual", json={**bad, "force": True}).status_code == 200


def test_thumbnail_is_small_jpeg(tmp_path: Path):
    path, _ = _images(tmp_path, 1)[0]
    with TestClient(app) as c:
        doc = c.post("/api/v1/documents", files={"file": (path.name, path.read_bytes(), "image/png")}).json()
        r = c.get(f"/api/v1/documents/{doc['document_id']}/image?variant=thumb")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert "max-age" in r.headers.get("cache-control", "")
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) <= 240


def test_huge_image_is_rejected_cleanly():
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".png", np.zeros((8000, 8000, 3), np.uint8))   # 64 ล้านพิกเซล แต่ไฟล์เล็ก
    with TestClient(app) as c:
        r = c.post("/api/v1/documents", files={"file": ("bomb.png", buf.tobytes(), "image/png")})
        assert r.status_code == 400
