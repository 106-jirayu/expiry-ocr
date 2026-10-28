"""ช่องทางส่งแจ้งเตือน — เปิดใช้เฉพาะช่องทางที่ตั้งค่าใน .env (ไม่ตั้ง = ข้าม)

- Telegram   : TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID  (สร้างบอทกับ @BotFather)
- Discord    : DISCORD_WEBHOOK_URL                    (Server Settings -> Integrations -> Webhooks)
- Webhook    : ALERT_WEBHOOK_URL  ส่ง JSON (ต่อ LINE Messaging API / n8n / Make / ระบบอื่นได้)
- Email      : SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO (Gmail ใช้ App Password)
หน้าเว็บมือถือแสดงแจ้งเตือน (และ browser notification) ได้เสมอโดยไม่ต้องตั้งค่า
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

import httpx

from app.config import Settings, settings

log = logging.getLogger(__name__)


def configured_channels(cfg: Settings = settings) -> list[str]:
    ch = []
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        ch.append("telegram")
    if cfg.discord_webhook_url:
        ch.append("discord")
    if cfg.alert_webhook_url:
        ch.append("webhook")
    if cfg.smtp_host and cfg.alert_email_to:
        ch.append("email")
    return ch


def send(title: str, text: str, payload: dict | None = None, cfg: Settings = settings) -> list[str]:
    """ส่งไปทุกช่องทางที่ตั้งค่าไว้ คืนรายชื่อช่องทางที่ส่งสำเร็จ (ช่องทางที่ล้มเหลวจะ log ไว้ ไม่ทำให้ระบบล่ม)"""
    sent = []
    for name in configured_channels(cfg):
        try:
            if name == "telegram":
                httpx.post(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage",
                           json={"chat_id": cfg.telegram_chat_id, "text": f"{title}\n{text}"},
                           timeout=10).raise_for_status()
            elif name == "discord":
                httpx.post(cfg.discord_webhook_url, json={"content": f"**{title}**\n{text}"},
                           timeout=10).raise_for_status()
            elif name == "webhook":
                httpx.post(cfg.alert_webhook_url, json={"title": title, "text": text, **(payload or {})},
                           timeout=10).raise_for_status()
            elif name == "email":
                msg = EmailMessage()
                msg["Subject"] = title
                msg["From"] = cfg.smtp_user or cfg.alert_email_to
                msg["To"] = cfg.alert_email_to
                msg.set_content(text)
                with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as s:
                    s.starttls()
                    if cfg.smtp_user:
                        s.login(cfg.smtp_user, cfg.smtp_password)
                    s.send_message(msg)
            sent.append(name)
        except Exception as e:  # noqa: BLE001 — ช่องทางใดล่มต้องไม่กระทบช่องทางอื่น
            log.warning("ส่งแจ้งเตือนทาง %s ไม่สำเร็จ: %s", name, e)
    return sent
