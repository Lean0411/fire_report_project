from flask import Flask
from app.config import Config
import os

def create_app(config_class=Config):
    app = Flask(__name__, 
                static_folder='../static',
                template_folder='../templates')
    app.config.from_object(config_class)
    
    # 確保必要的目錄存在
    for directory in [app.config['UPLOAD_FOLDER'], 
                     os.path.dirname(app.config['MODEL_PATH']), 
                     app.config['LOG_FOLDER']]:
        os.makedirs(directory, exist_ok=True)
    
    # 註冊路由藍圖
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app