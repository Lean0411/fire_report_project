"""
應用配置設定模組
包含Flask應用配置、環境變數載入和基本設定
"""
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

class Config:
    """應用程式配置類"""
    
    # Flask 基本設定
    UPLOAD_FOLDER = 'static/uploads/'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB 上限
    SEND_FILE_MAX_AGE_DEFAULT = 0  # 開發模式下停用快取
    
    # OpenAI 設定
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Ollama 設定
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma:7b')
    
    # 路徑設定
    KNOWLEDGE_BASE_PATH = 'knowledge_base/sop.json'
    MODEL_PATH = 'models/fire_detection/deep_wildfire_cnn_model_amp.pth'
    LOG_DIR = 'logs'
    
    # 必要的目錄列表
    REQUIRED_DIRECTORIES = [
        UPLOAD_FOLDER,
        'models/fire_detection',
        'static',
        LOG_DIR
    ]
    
    @classmethod
    def init_app(cls, app):
        """初始化Flask應用配置"""
        app.config.update(
            UPLOAD_FOLDER=cls.UPLOAD_FOLDER,
            ALLOWED_EXTENSIONS=cls.ALLOWED_EXTENSIONS,
            MAX_CONTENT_LENGTH=cls.MAX_CONTENT_LENGTH,
            SEND_FILE_MAX_AGE_DEFAULT=cls.SEND_FILE_MAX_AGE_DEFAULT
        )
        
        # 確保必要的目錄存在
        for directory in cls.REQUIRED_DIRECTORIES:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_openai_config(cls):
        """獲取OpenAI配置"""
        return {
            'api_key': cls.OPENAI_API_KEY
        }
    
    @classmethod
    def get_ollama_config(cls):
        """獲取Ollama配置"""
        return {
            'host': cls.OLLAMA_HOST,
            'model': cls.OLLAMA_MODEL
        }