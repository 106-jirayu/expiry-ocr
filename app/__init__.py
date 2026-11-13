"""Expiry Date AI OCR"""
import os

# กันภาพ "decompression bomb" (ไฟล์เล็กแต่ขนาดพิกเซลมหาศาล) กินหน่วยความจำเซิร์ฟเวอร์ — ต้องตั้งก่อน import cv2
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(50_000_000))
