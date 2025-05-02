import os
from datetime import datetime
from app.config import Config
import logging

# 設定日誌
def setup_logging():
    """設定應用程式日誌"""
    os.makedirs(Config.LOG_FOLDER, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

def allowed_file(filename):
    """檢查檔案是否為允許的格式"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def generate_timestamp():
    """產生時間戳記，用於檔名"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_file_extension(filename):
    """取得檔案副檔名"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''