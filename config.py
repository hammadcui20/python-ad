# config.py
import os

class AppConfig:
    DEBUG = False
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB uploads
    UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/flask_ad_detector_uploads")