"""การเชื่อมต่อฐานข้อมูล (SQLite ค่าเริ่มต้น / PostgreSQL ผ่าน DATABASE_URL)"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def make_engine(url: str):
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        db_path = url.replace("sqlite:///", "", 1)
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True, **kwargs)


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(bind=None) -> None:
    from app import models  # noqa: F401  (ลงทะเบียนตาราง)

    bind = bind or engine
    Base.metadata.create_all(bind=bind)
    _add_missing_columns(bind)


# คอลัมน์ที่เพิ่มทีหลัง: create_all ไม่แก้ตารางที่มีอยู่แล้ว จึงเติมให้เองเพื่อให้ฐานข้อมูลเดิมใช้ต่อได้
_LATE_COLUMNS = {
    "documents": {"product_name": "VARCHAR(200)", "removed_at": "TIMESTAMP", "removed_by": "VARCHAR(128)",
                  "removed_note": "VARCHAR(200)"},
}


def _add_missing_columns(bind) -> None:
    insp = inspect(bind)
    with bind.begin() as conn:
        for table, cols in _LATE_COLUMNS.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
