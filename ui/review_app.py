"""HITL Review UI (Streamlit) — หน้าจอสำหรับเจ้าหน้าที่ QA บนคอมพิวเตอร์

รัน:  streamlit run ui/review_app.py   (ต้องเปิด API ก่อน: uvicorn app.main:app)

1. Layout Display      : ภาพต้นฉบับ + กรอบไฮไลท์, ค่าที่ระบบทำนาย, คะแนนความมั่นใจ, เหตุผลที่เข้าคิว
2. Verification        : ปุ่ม "อนุมัติ (Confirm)" และช่อง "แก้ไขข้อความ (Edit)" ต่อฟิลด์
3. Audit Trail Logging : Who / When / Original / Corrected / Reason (แสดงในแท็บ Audit)
4. Feedback Data       : ทุกการตรวจถูกบันทึกเป็นคู่ภาพ-ข้อความ ดาวน์โหลดเป็น dataset ได้
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/") + "/api/v1"
FIELD_TH = {"exp_date": "วันหมดอายุ (EXP)", "mfg_date": "วันผลิต (MFG)", "lot_number": "Lot Number"}
STATUS_TH = {"AUTO_APPROVED": "🟢 อนุมัติอัตโนมัติ", "PENDING_REVIEW": "🟠 รอตรวจ (HITL)",
             "HUMAN_APPROVED": "🟢 คนยืนยัน", "HUMAN_CORRECTED": "🔵 คนแก้ไข", "REJECTED": "🔴 ปฏิเสธ"}

st.set_page_config(page_title="HITL Review — Expiry OCR", page_icon="📦", layout="wide")
client = httpx.Client(timeout=60)


def api(method: str, path: str, **kw) -> tuple[Any, Any]:
    r = client.request(method, API + path, **kw)
    if r.status_code >= 400:
        try:
            return None, r.json().get("detail")
        except Exception:
            return None, r.text
    return (r.json() if "json" in r.headers.get("content-type", "") else r.content), None


def conf_badge(c: float, th: float = 0.9) -> str:
    color = "green" if c >= th else ("orange" if c >= 0.7 else "red")
    return f":{color}[**{c:.1%}**]"


def reason_th(r: str, rules: list[dict]) -> str:
    if r.startswith("LOW_CONFIDENCE:"):
        f, v = r[15:].split("=", 1)
        return f"ความมั่นใจต่ำ — {FIELD_TH.get(f, f)} {v}"
    if r.startswith("ENSEMBLE_DISAGREE:"):
        return f"อ่านหลายรอบไม่ตรงกัน — {FIELD_TH.get(r[18:].split()[0], r)}"
    if r.startswith("RULE_FAIL:"):
        rid, _, f = r[10:].split(":")
        msg = next((x["message"] for x in rules if x["rule_id"] == rid and x["field"] == f and not x["passed"]), f)
        return f"ไม่ผ่านกฎ {rid} — {msg}"
    return r


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📦 Expiry OCR — HITL")
    reviewer = st.text_input("ชื่อผู้ตรวจ (Reviewer)", value=st.session_state.get("reviewer", ""))
    reviewer = reviewer or ""
    st.session_state["reviewer"] = reviewer
    up = st.file_uploader("อัปโหลดภาพใหม่", type=["png", "jpg", "jpeg", "webp", "bmp"], accept_multiple_files=True)
    if up and st.button("ประมวลผล", width="stretch"):
        for f in up:
            data, err = api("POST", "/documents", files={"file": (f.name, f.getvalue(), f.type)})
            st.write(f"{f.name}: {STATUS_TH.get(data['status']) if data else err}")
    m, _ = api("GET", "/metrics")
    if m:
        st.divider()
        st.metric("ภาพทั้งหมด", m["total_documents"])
        c1, c2 = st.columns(2)
        c1.metric("First-Pass Yield", f"{(m['first_pass_yield'] or 0):.1%}")
        c2.metric("Exception Rate", f"{(m['exception_rate'] or 0):.1%}")
        st.metric("ค้างในคิว", m["by_status"]["PENDING_REVIEW"])

tab_queue, tab_alerts, tab_all, tab_audit, tab_fb = st.tabs(
    ["🗂️ คิวตรวจ", "⏰ แจ้งเตือนวันหมดอายุ", "📋 เอกสารทั้งหมด", "🧾 Audit Trail", "🔁 Feedback Data"])

# ------------------------------------------------------------------ queue
with tab_queue:
    order = st.radio("ลำดับ", ["fifo", "hardest"], horizontal=True,
                     format_func=lambda x: "เข้าก่อน-ตรวจก่อน" if x == "fifo" else "ยากสุดก่อน (มั่นใจต่ำสุด)")
    queue, err = api("GET", "/review/queue", params={"limit": 200, "order": order})
    if err:
        st.error(f"เชื่อมต่อ API ไม่ได้: {err}")
    elif not queue:
        st.success("🎉 ไม่มีงานค้างในคิว")
    else:
        ids = [d["document_id"] for d in queue]
        idx = st.session_state.get("q_idx", 0) % len(ids)
        doc_id = st.selectbox("เลือกเอกสาร", ids, index=idx,
                              format_func=lambda i: next(f"#{d['document_id']} · {d['filename']} · "
                                                         f"มั่นใจ {d['overall_confidence']:.0%}" for d in queue
                                                         if d["document_id"] == i))
        doc, derr = api("GET", f"/documents/{doc_id}")
        if not doc:
            st.error(f"โหลดเอกสารไม่ได้: {derr}")
            st.stop()
        left, right = st.columns([3, 2], gap="large")
        with left:
            variant = st.radio("ภาพ", ["annotated", "original", "processed"], horizontal=True,
                               format_func={"annotated": "ต้นฉบับ + กรอบไฮไลท์", "original": "ต้นฉบับ",
                                            "processed": "หลัง preprocessing"}.get)
            img, _ = api("GET", f"/documents/{doc_id}/image", params={"variant": variant})
            if img:
                st.image(img, width="stretch")
            st.caption("Preprocessing: " + ", ".join(doc["preprocessing"].get("steps", [])))
            with st.expander("ข้อความ OCR ทุกรอบ (ensemble)"):
                st.json(doc["preprocessing"].get("pass_fields"))
        with right:
            st.subheader(f"เอกสาร #{doc_id}")
            st.markdown("**เหตุผลที่ส่งเข้าคิว**")
            for r in doc["route_reasons"]:
                st.markdown(f"- {reason_th(r, doc['rules'])}")
            with st.form(f"review_{doc_id}"):
                edits = {}
                for name in ("exp_date", "mfg_date", "lot_number"):
                    f = doc["fields"].get(name) or {}
                    pv = f.get("predicted_value")
                    st.markdown(f"**{FIELD_TH[name]}** — ทำนาย: `{pv or 'ไม่พบ'}` "
                                f"(อ่านได้ `{f.get('predicted_raw') or '-'}`) ความมั่นใจ {conf_badge(f.get('confidence', 0))}")
                    if name == "lot_number":
                        edits[name] = st.text_input("แก้ไข Lot", value=pv or "", key=f"{doc_id}_{name}",
                                                    label_visibility="collapsed")
                    else:
                        c1, c2 = st.columns([2, 1])
                        default = date.fromisoformat(pv) if pv else None
                        v = c1.date_input("แก้ไขวันที่", value=default, key=f"{doc_id}_{name}", format="DD/MM/YYYY",
                                          label_visibility="collapsed")
                        none = c2.checkbox("ไม่มีบนฉลาก", value=pv is None, key=f"{doc_id}_{name}_none")
                        edits[name] = None if none or v is None else v.isoformat()
                raw_fix = st.text_input("ข้อความ EXP ตามที่พิมพ์บนฉลาก (ไม่บังคับ ใช้สอนโมเดล)",
                                        placeholder="เช่น 31/12/2026")
                reason = st.text_input("เหตุผลการแก้ไข", placeholder="เช่น OCR อ่าน 8 เป็น 3 เพราะภาพเบลอ")
                force = st.checkbox("ยืนยันแม้ค่าไม่ผ่านกฎธุรกิจ (ต้องใส่เหตุผล)")
                c1, c2 = st.columns(2)
                confirm = c1.form_submit_button("✅ อนุมัติ / บันทึก (Confirm)", type="primary", width="stretch")
                reject = c2.form_submit_button("⛔ ปฏิเสธภาพ", width="stretch")
            if confirm or reject:
                if not reviewer.strip():
                    st.error("กรุณาใส่ชื่อผู้ตรวจที่แถบซ้าย")
                elif reject:
                    _, err = api("POST", f"/review/{doc_id}/reject",
                                 json={"reviewer": reviewer, "reason": reason or "ภาพใช้ไม่ได้"})
                    st.toast("ปฏิเสธแล้ว") if not err else st.error(err)
                    if not err:
                        st.rerun()
                else:
                    corrections = []
                    for name, v in edits.items():
                        pv = (doc["fields"].get(name) or {}).get("predicted_value")
                        v = (v or "").strip().upper() if name == "lot_number" else v
                        if (v or None) != pv:
                            corrections.append({"name": name, "value": v or None, "reason": reason or None,
                                                "raw_text": raw_fix or None if name == "exp_date" else None})
                    _, err = api("POST", f"/review/{doc_id}/submit",
                                 json={"reviewer": reviewer, "corrections": corrections, "force": force})
                    if err:
                        st.error(err.get("message") if isinstance(err, dict) else err)
                        for d in (err.get("details") if isinstance(err, dict) else []) or []:
                            st.warning(f"{d['rule_id']}: {d['message']}")
                    else:
                        st.toast("บันทึกแล้ว ✓" + (" (เก็บเป็น feedback data)" if corrections else ""))
                        st.rerun()

# ------------------------------------------------------------------ alerts
with tab_alerts:
    KIND_TH = {"EXPIRED": "⛔ หมดอายุแล้ว", "EXPIRES_TODAY": "🚨 หมดอายุวันนี้", "EXPIRING_SOON": "⏰ ใกล้หมดอายุ"}
    c1, c2 = st.columns([1, 3])
    if c1.button("ตรวจวันหมดอายุเดี๋ยวนี้"):
        res, _ = api("POST", "/alerts/check")
        if res:
            st.toast(f"แจ้งเตือนใหม่ {len(res['created'])} รายการ · ช่องทาง: {', '.join(res['channels']) or 'หน้าเว็บเท่านั้น'}")
    show_all = c2.checkbox("แสดงรายการที่รับทราบแล้วด้วย")
    al, _ = api("GET", "/alerts", params={"include_acknowledged": show_all})
    if not al:
        st.success("ไม่มีสินค้าใกล้หมดอายุ")
    for a in al or []:
        with st.container(border=True):
            l, r = st.columns([4, 1])
            l.markdown(f"**{KIND_TH.get(a['kind'], a['kind'])}** · EXP `{a['exp_date']}` · Lot `{a['lot_number'] or '-'}` "
                       f"· เอกสาร #{a['document_id']}")
            l.caption(a["message"].replace("\n", " · ") + (f" · รับทราบโดย {a['acknowledged_by']}" if a["acknowledged"] else ""))
            if not a["acknowledged"] and r.button("รับทราบ", key=f"ack_{a['id']}"):
                if not reviewer.strip():
                    st.error("กรุณาใส่ชื่อผู้ตรวจที่แถบซ้าย")
                else:
                    api("POST", f"/alerts/{a['id']}/ack", json={"user": reviewer})
                    st.rerun()

# ------------------------------------------------------------------ all documents
with tab_all:
    status = st.selectbox("สถานะ", ["ทั้งหมด"] + list(STATUS_TH))
    docs, _ = api("GET", "/documents", params={"limit": 500, **({"status": status} if status != "ทั้งหมด" else {})})
    rows = []
    for d in docs or []:
        row = {"id": d["document_id"], "ไฟล์": d["filename"], "สถานะ": STATUS_TH.get(d["status"]),
               "ความมั่นใจ": d["overall_confidence"]}
        for n in ("exp_date", "mfg_date", "lot_number"):
            f = d["fields"].get(n) or {}
            row[n] = f.get("value") or f.get("predicted_value")
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"ความมั่นใจ": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    exp, _ = api("GET", "/export/documents")
    if exp:
        import json
        st.download_button("⬇️ ส่งออก JSON (downstream)", json.dumps(exp, ensure_ascii=False, indent=2),
                           "expiry_export.json", "application/json")

# ------------------------------------------------------------------ audit
with tab_audit:
    did = st.number_input("เลขเอกสาร", min_value=1, step=1)
    audit, err = api("GET", f"/documents/{int(did)}/audit")
    if audit:
        audit_df = pd.DataFrame(audit)[["when", "who", "action", "field", "original_value", "corrected_value", "reason"]]
        st.dataframe(audit_df.set_axis(["เมื่อไร", "ใคร", "การกระทำ", "ฟิลด์", "ค่าเดิม", "ค่าใหม่", "เหตุผล"], axis=1),
                     width="stretch", hide_index=True)
    elif err:
        st.info("ไม่พบเอกสาร")

# ------------------------------------------------------------------ feedback
with tab_fb:
    st.markdown("ทุกฟิลด์ที่คนตรวจแล้วถูกบันทึกเป็น **คู่ภาพ crop + ข้อความที่ถูกต้อง** "
                "ใช้ fine-tune TrOCR ด้วย `python -m scripts.finetune_trocr --data <โฟลเดอร์>`")
    if m:
        st.json({"per_field": m["per_field"], "field_correction_rate": m["field_correction_rate"]})
    only_err = st.checkbox("เฉพาะเคสที่ OCR อ่านผิด")
    if st.button("เตรียมไฟล์ dataset"):
        z, err = api("GET", "/export/feedback", params={"only_errors": only_err})
        if z:
            st.download_button("⬇️ ดาวน์โหลด feedback_dataset.zip", z, "feedback_dataset.zip", "application/zip")
