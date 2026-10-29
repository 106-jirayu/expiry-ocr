from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.alerts.service import classify
from app.main import app
from app.models import AlertKind
from scripts.generate_synthetic import generate

TODAY = date(2026, 10, 28)


def test_classify():
    assert classify(TODAY - timedelta(days=1), TODAY, 7) == AlertKind.EXPIRED
    assert classify(TODAY, TODAY, 7) == AlertKind.EXPIRES_TODAY
    assert classify(TODAY + timedelta(days=7), TODAY, 7) == AlertKind.EXPIRING_SOON
    assert classify(TODAY + timedelta(days=8), TODAY, 7) is None


def test_alert_created_once_and_acknowledged(tmp_path):
    generate(tmp_path, 1, seed=99)
    img = next((tmp_path / "images").glob("*.png"))
    soon = (date.today() + timedelta(days=3)).isoformat()
    with TestClient(app) as c:
        doc = c.post("/api/v1/documents", files={"file": (img.name, img.read_bytes(), "image/png")}).json()
        r = c.post(f"/api/v1/review/{doc['document_id']}/submit", json={
            "reviewer": "qa1", "force": True,
            "corrections": [{"name": "exp_date", "value": soon}, {"name": "mfg_date", "value": None}]})
        assert r.status_code == 200, r.text
        assert r.json()["expiry"] == {"exp_date": soon, "days_left": 3, "status": "EXPIRING_SOON"}

        mine = [a for a in c.get("/api/v1/alerts").json() if a["document_id"] == doc["document_id"]]
        assert len(mine) == 1 and mine[0]["kind"] == "EXPIRING_SOON" and "เหลืออีก 3 วัน" in mine[0]["message"]
        c.post("/api/v1/alerts/check")  # ตรวจซ้ำต้องไม่สร้างแจ้งเตือนซ้ำ
        assert len([a for a in c.get("/api/v1/alerts").json() if a["document_id"] == doc["document_id"]]) == 1

        ack = c.post(f"/api/v1/alerts/{mine[0]['id']}/ack", json={"user": "store1"}).json()
        assert ack["acknowledged"] and ack["acknowledged_by"] == "store1"
        assert all(a["id"] != mine[0]["id"] for a in c.get("/api/v1/alerts").json())
        assert c.post("/api/v1/alerts/test").json() == {"configured": [], "sent": []}


def test_calendar_lists_approved_expiry_dates(tmp_path):
    generate(tmp_path, 1, seed=98)
    img = next((tmp_path / "images").glob("*.png"))
    exp = (date.today() + timedelta(days=20)).isoformat()
    with TestClient(app) as c:
        doc = c.post("/api/v1/documents", files={"file": (img.name, img.read_bytes(), "image/png")}).json()
        c.post(f"/api/v1/review/{doc['document_id']}/submit", json={
            "reviewer": "qa1", "force": True, "corrections": [{"name": "exp_date", "value": exp}]})
        rows = c.get("/api/v1/calendar", params={"start": exp, "end": exp}).json()
        mine = [r for r in rows if r["document_id"] == doc["document_id"]]
        assert len(mine) == 1 and mine[0]["exp_date"] == exp and mine[0]["status"] == "OK"
        assert mine[0]["days_left"] == 20
        assert c.get("/api/v1/calendar", params={"start": exp, "end": "2000-01-01"}).status_code == 400
