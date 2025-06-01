import os
import logging
from flask import Flask
from app.config import Config

def create_app(config_class=Config):
    # 取得 app 目錄的絕對路徑
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir))

    # 建立 Flask 實例，指定靜態與樣板路徑
    app = Flask(
        __name__,
        static_folder=os.path.join(project_root, 'static'),
        template_folder=os.path.join(project_root, 'templates')
    )

    # 載入配置
    app.config.from_object(config_class)

    # ==== 日誌設定 ====
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
    logging.basicConfig(
        filename=app.config['LOG_FILE'],
        format=app.config['LOG_FORMAT'],
        level=getattr(logging, app.config['LOG_LEVEL'])
    )
    app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))

    # 確保關鍵資料夾存在
    for folder in [
        app.config['UPLOAD_FOLDER'],
        os.path.dirname(app.config['MODEL_PATH']),
        app.config['LOG_FOLDER']
    ]:
        os.makedirs(folder, exist_ok=True)

    # 註冊路由藍圖
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
