"""ตัวเปิดระบบแบบคลิกเดียว (ใช้ได้ทั้ง Windows / macOS / Linux)

    python run.py              # เปิด API + หน้าเว็บมือถือ + Streamlit แล้วเปิดเบราว์เซอร์ให้
    python run.py --no-ui      # เปิดเฉพาะ API + หน้าเว็บมือถือ
    python run.py --port 8080

ครั้งแรกจะสร้างภาพตัวอย่าง 30 ภาพที่ data/synthetic/images ไว้ทดลองอัปโหลด
กด Ctrl+C เพื่อปิดทั้งหมด
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def check_deps() -> None:
    missing = []
    for mod in ("fastapi", "uvicorn", "sqlalchemy", "cv2", "PIL", "numpy", "multipart", "rapidocr_onnxruntime"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"ยังไม่ได้ติดตั้งแพ็กเกจ: {', '.join(missing)}")
        print(f"รัน:  {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)


def wait_until_up(url: str, timeout: float = 60) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    ap.add_argument("--no-ui", action="store_true", help="ไม่เปิด Streamlit")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    check_deps()
    samples = ROOT / "data" / "synthetic"
    if not (samples / "manifest.jsonl").exists():
        print("สร้างภาพตัวอย่าง 30 ภาพ ...")
        subprocess.run([sys.executable, "-m", "scripts.generate_synthetic", "--out", str(samples), "--n", "30"],
                       cwd=ROOT, check=True)

    procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0",
                               "--port", str(a.port)], cwd=ROOT)]
    base = f"http://localhost:{a.port}"
    if not wait_until_up(base + "/health"):
        print("เปิด API ไม่สำเร็จ ดูข้อความ error ด้านบน")
        for p in procs:
            p.terminate()
        sys.exit(1)

    has_streamlit = True
    try:
        __import__("streamlit")
    except ImportError:
        has_streamlit = False
    if not a.no_ui and has_streamlit:
        procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/review_app.py",
                                       "--server.port", str(a.ui_port), "--server.headless", "true"], cwd=ROOT))

    ip = lan_ip()
    print("\n" + "=" * 64)
    print(f"  หน้าเว็บ (คอมพิวเตอร์) : {base}/")
    print(f"  หน้าเว็บ (มือถือ Wi-Fi เดียวกัน): http://{ip}:{a.port}/")
    print(f"  API docs (Swagger)   : {base}/docs")
    if len(procs) > 1:
        print(f"  HITL Review (QA)     : http://localhost:{a.ui_port}/")
    print(f"  ภาพตัวอย่างทดลองอัปโหลด: {samples / 'images'}")
    print("  กด Ctrl+C เพื่อปิด")
    print("=" * 64 + "\n")
    if not a.no_browser:
        webbrowser.open(base + "/")

    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
