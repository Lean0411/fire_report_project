import os
from dotenv import load_dotenv

# 載入根目錄下的 .env 檔案
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(basedir, os.pardir))
load_dotenv(os.path.join(project_root, '.env'))

class Config:
    # ===== 基礎路徑 =====
    BASE_DIR = project_root

    # ===== 上傳相關 =====
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    SEND_FILE_MAX_AGE_DEFAULT = 0  # 開發模式下停用快取

    # ===== 模型相關 =====
    MODEL_PATH = os.getenv(
        'MODEL_PATH',
        os.path.join(BASE_DIR, 'models', 'fire_detection', 'deep_wildfire_cnn_model_amp.pth')
    )
    IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 224))
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))

    # ===== LLM (Ollama) 設定 =====
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma:7b')
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 512))
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.3))

    # ===== 日誌設定 =====
    LOG_FOLDER = os.getenv(
        'LOG_FOLDER',
        os.path.join(BASE_DIR, 'logs')
    )
    LOG_FILE = os.path.join(LOG_FOLDER, 'app.log')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

class DevelopmentConfig(Config):
    DEBUG = True
    ENV = 'development'

class ProductionConfig(Config):
    DEBUG = False
    ENV = 'production'
    # 生產環境長時間快取
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1年
    LOG_LEVEL = 'ERROR'

# 根據 FLASK_ENV 自動選擇配置
def get_config():
    env = os.getenv('FLASK_ENV', 'development').lower()
    if env == 'production':
        return ProductionConfig
    return DevelopmentConfig
