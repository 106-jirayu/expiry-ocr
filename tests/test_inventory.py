from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from app.db import init_db
from app.inventory import bucket_of
from app.main import app


def _manual(c, name, days, lot=None):
    exp = (date.today() + timedelta(days=days)).isoformat()
    r = c.post("/api/v1/documents/manual", json={"user": "qa", "product_name": name, "exp_date": exp,
                                                 "lot_number": lot, "force": True})
    assert r.status_code == 200, r.text
    return r.json()["document_id"]


def test_bucket_of():
    assert [bucket_of(d, 7) for d in (-1, 0, 1, 7, 8, 30, 31)] == \
        ["expired", "today", "soon", "soon", "month", "month", "later"]


def test_inventory_search_filter_remove_restore_and_csv():
    with TestClient(app) as c:
        milk = _manual(c, "นมกล่อง INV-TEST", 3, "LOT77")
        bread = _manual(c, "ขนมปัง INV-TEST", -2)
        rice = _manual(c, "ข้าวสาร INV-TEST", 90)

        inv = c.get("/api/v1/inventory", params={"q": "inv-test"}).json()
        assert [i["document_id"] for i in inv["items"]] == [bread, milk, rice]  # ใกล้หมดอายุก่อน
        assert inv["summary"]["expired"] == 1 and inv["summary"]["soon"] == 1 and inv["summary"]["later"] == 1
        assert inv["summary"]["in_stock"] == 3
        assert c.get("/api/v1/inventory", params={"q": "lot77"}).json()["items"][0]["document_id"] == milk
        assert c.get("/api/v1/inventory", params={"q": f"#{rice}"}).json()["items"][0]["document_id"] == rice
        only = c.get("/api/v1/inventory", params={"q": "INV-TEST", "bucket": "expired"}).json()["items"]
        assert [i["document_id"] for i in only] == [bread]

        # นำออกจากชั้น -> ปิดแจ้งเตือนที่ค้าง และไม่นับในคลัง
        assert any(a["document_id"] == bread for a in c.get("/api/v1/alerts").json())
        r = c.post(f"/api/v1/documents/{bread}/remove", json={"user": "store1", "note": "ทิ้งแล้ว"})
        assert r.status_code == 200 and r.json()["removed"]["by"] == "store1"
        assert all(a["document_id"] != bread for a in c.get("/api/v1/alerts").json())
        inv = c.get("/api/v1/inventory", params={"q": "inv-test"}).json()
        assert bread not in [i["document_id"] for i in inv["items"]] and inv["summary"]["removed"] == 1
        assert all(r["document_id"] != bread for r in c.get(
            "/api/v1/calendar", params={"start": "2000-01-01", "end": "2000-12-31"}).json())
        removed = c.get("/api/v1/inventory", params={"q": "inv-test", "bucket": "removed"}).json()["items"]
        assert removed[0]["removed"]["note"] == "ทิ้งแล้ว"
        c.post(f"/api/v1/documents/{bread}/restore", json={"user": "store1"})
        assert c.get("/api/v1/inventory", params={"q": "inv-test"}).json()["summary"]["expired"] == 1

        # ตั้งชื่อสินค้า + audit
        doc = c.patch(f"/api/v1/documents/{rice}/product", json={"user": "qa", "product_name": "ข้าวหอมมะลิ INV-TEST"}).json()
        assert doc["product_name"] == "ข้าวหอมมะลิ INV-TEST"
        assert doc["audit_trail"][-1]["field"] == "product_name"

        csv = c.get("/api/v1/export/inventory.csv", params={"q": "inv-test"})
        assert csv.status_code == 200 and csv.content.startswith(b"\xef\xbb\xbf")
        text_ = csv.content.decode("utf-8-sig")
        assert text_.splitlines()[0].startswith("รหัส,ชื่อสินค้า,วันหมดอายุ")
        assert "ข้าวหอมมะลิ INV-TEST" in text_ and "หมดอายุแล้ว" in text_


def test_remove_requires_reviewed_document(tmp_path):
    from scripts.generate_synthetic import generate
    generate(tmp_path, 1, seed=5)
    img = next((tmp_path / "images").glob("*.png"))
    with TestClient(app) as c:
        doc = c.post("/api/v1/documents", files={"file": (img.name, img.read_bytes(), "image/png")}).json()
        if doc["status"] == "PENDING_REVIEW":
            assert c.post(f"/api/v1/documents/{doc['document_id']}/remove", json={"user": "x"}).status_code == 409


def test_init_db_adds_columns_to_old_database(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/old.db")
    with eng.begin() as conn:  # ตาราง documents แบบเวอร์ชันก่อน (ไม่มีคอลัมน์คลังสินค้า)
        conn.execute(text("CREATE TABLE documents (id INTEGER PRIMARY KEY, filename VARCHAR(255))"))
    init_db(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("documents")}
    assert {"product_name", "removed_at", "removed_by", "removed_note"} <= cols
