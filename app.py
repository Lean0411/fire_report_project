"""
火災檢測系統主應用程式
使用模組化架構重構後的Flask應用
"""
from flask import Flask, render_template, request, redirect

# 導入配置和日誌
from config.settings import Config
from config.logging_config import setup_logging

# 導入API藍圖
from api.detection import detection_bp
from api.safety import safety_bp

# 導入工具和常數
from utils.constants import CATEGORY_LABELS
from config.logging_config import get_logger

def create_app():
    """創建Flask應用實例"""
    app = Flask(__name__)
    
    # 初始化配置
    Config.init_app(app)
    
    # 設置日誌
    setup_logging(app)
    
    logger = get_logger(__name__)
    logger.info("火災檢測系統啟動中...")
    
    # 註冊藍圖
    app.register_blueprint(detection_bp)
    app.register_blueprint(safety_bp)
    
    # 靜態資源路由
    @app.route('/favicon.ico')
    def favicon():
        return app.send_static_file('favicon.ico')
    
    @app.route('/apple-touch-icon.png')
    @app.route('/apple-touch-icon-precomposed.png') 
    def apple_touch_icon():
        return app.send_static_file('apple-touch-icon.png')
    
    # 主頁路由
    @app.route('/', methods=['GET', 'POST'])
    def index():
        """主頁路由"""
        if request.method == 'POST':
            # POST請求重定向到檢測API
            return redirect('/api/detect', code=307)
        
        # GET請求顯示主頁
        return render_template('index.html', 
                             category_labels=CATEGORY_LABELS)
    
    logger.info("Flask應用初始化完成")
    return app

# 創建應用實例
app = create_app()

if __name__ == '__main__':
    # 開發環境運行
    import os
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)