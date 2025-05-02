# app/__init__.py
import os
from flask import Flask
from app.config import Config

def create_app(config_class=Config):
    # 取得 app 目錄的絕對路徑
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # 預設把根目錄當成 flask 的 project root
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir))

    app = Flask(
        __name__,
        static_folder=os.path.join(project_root, 'static'),
        template_folder=os.path.join(project_root, 'templates')
    )
    app.config.from_object(config_class)

    # 確保要用到的資料夾都存在
    for d in [
        app.config['UPLOAD_FOLDER'], 
        os.path.dirname(app.config['MODEL_PATH']), 
        app.config['LOG_FOLDER']
    ]:
        os.makedirs(d, exist_ok=True)

    # 註冊藍圖
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
