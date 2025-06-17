"""
日誌配置模組
提供統一的日誌設定功能
"""
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from .settings import Config

def setup_logging(app):
    """
    設定應用程式日誌系統
    
    Args:
        app: Flask應用實例
    """
    # 確保日誌目錄存在
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 設定日誌格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    
    # 檔案處理器 (每個檔案最大 10MB，保留最後 5 個檔案)
    file_handler = RotatingFileHandler(
        os.path.join(Config.LOG_DIR, 'app.log'),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    # 設定 Flask logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)
    
    # 設定 Werkzeug logger
    logging.getLogger('werkzeug').addHandler(file_handler)
    
    app.logger.info("日誌系統初始化完成")

def get_logger(name=None):
    """
    獲取日誌記錄器
    
    Args:
        name: 記錄器名稱，預設為None
        
    Returns:
        Logger: 日誌記錄器實例
    """
    return logging.getLogger(name)