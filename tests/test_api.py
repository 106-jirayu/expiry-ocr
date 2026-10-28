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
        docs = []
        for path, gt in _images(tmp_path):
            r = c.post("/api/v1/documents", files={"file": (path.name, path.read_bytes(), "image/png")})
            assert r.status_code == 200, r.text
            docs.append((r.json(), gt))
        statuses = {d["status"] for d, _ in docs}
        assert statuses <= {"AUTO_APPROVED", "PENDING_REVIEW"}

        queue = c.get("/api/v1/review/queue").json()
        assert len(queue) == sum(d["status"] == "PENDING_REVIEW" for d, _ in docs)
        if not queue:
            return
        doc_id = queue[0]["document_id"]
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
        assert m["total_documents"] == len(docs)
        assert abs(m["first_pass_yield"] + m["exception_rate"] - 1) < 1e-9
        fb = c.get("/api/v1/export/feedback")
        assert fb.status_code == 200 and fb.headers["content-type"] == "application/zip"
        assert isinstance(c.get("/api/v1/export/documents").json(), list)


def test_reject_and_bad_upload():
    with TestClient(app) as c:
        assert c.post("/api/v1/documents", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400
        assert c.post("/api/v1/review/99999/reject", json={"reviewer": "a", "reason": "b"}).status_code == 404
