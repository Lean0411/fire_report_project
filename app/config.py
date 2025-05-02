import os

class Config:
    # 基礎路徑
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 基本設定
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads/')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 上限
    SEND_FILE_MAX_AGE_DEFAULT = 0  # 開發模式下停用快取
    
    # 模型相關設定
    MODEL_PATH = os.path.join(BASE_DIR, 'models/fire_detection/deep_wildfire_cnn_model_amp.pth')
    IMAGE_SIZE = 224
    CONFIDENCE_THRESHOLD = 0.5
    
    # 日誌設定
    LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
    LOG_FILE = os.path.join(LOG_FOLDER, 'app.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    # 生產環境特定設定
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1年
    LOG_LEVEL = 'ERROR'