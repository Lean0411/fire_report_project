"""
FireGuard AI - 火災檢測系統主應用程式
使用模組化架構重構後的Flask應用，整合了完整的企業級功能
"""
import os
# 強制 PyTorch 使用 CPU，避免 CUDA 驅動問題
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA'] = '0'

from flask import Flask, render_template, request, redirect, jsonify
from flask_cors import CORS
import atexit

# 導入配置和日誌
from config.settings import Config
from config.logging_config import setup_logging, get_logger

# 導入數據庫
from data.database import init_db

# 導入認證服務
from services.auth.auth_service import auth_service

# 導入快取服務
from services.cache_service import cache_service

# 導入監控
from monitoring.health_checks import health_checker

# 導入 v1 API 藍圖
from api.v1.auth import auth_bp
from api.v1.detection import detection_bp
from api.v1.monitoring import monitoring_bp

# 導入舊版API藍圖（向後兼容）
from api.detection import detection_bp as legacy_detection_bp
from api.safety import safety_bp as legacy_safety_bp

# 導入中介軟體
from api.middleware.metrics import track_api_metrics

# 導入工具和常數
from utils.constants import CATEGORY_LABELS

def create_app(config_name='default'):
    """創建Flask應用實例"""
    app = Flask(__name__)
    
    # 初始化配置
    Config.init_app(app)
    
    # 設置日誌
    setup_logging(app)
    logger = get_logger(__name__)
    logger.info("FireGuard AI 系統啟動中...")
    
    # 初始化 CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:5000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-API-Key", "X-Session-ID"]
        }
    })
    
    # 初始化數據庫
    logger.info("初始化數據庫...")
    try:
        db = init_db(app)
        logger.info("數據庫初始化完成")
    except Exception as e:
        logger.error(f"數據庫初始化失敗: {str(e)}")
        raise
    
    # 初始化認證服務
    logger.info("初始化認證服務...")
    try:
        auth_service.init_app(app)
        logger.info("認證服務初始化完成")
    except Exception as e:
        logger.error(f"認證服務初始化失敗: {str(e)}")
        raise
    
    # 初始化快取服務
    logger.info("初始化快取服務...")
    try:
        redis_url = app.config.get('REDIS_URL', os.environ.get('REDIS_URL'))
        cache_service.__init__(redis_url)
        logger.info("快取服務初始化完成")
    except Exception as e:
        logger.warning(f"快取服務初始化警告: {str(e)}")
    
    # 註冊 API v1 藍圖
    logger.info("註冊 API v1 藍圖...")
    app.register_blueprint(auth_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(monitoring_bp)
    logger.info("API v1 藍圖註冊完成")
    
    # 註冊舊版API藍圖（向後兼容）
    logger.info("註冊舊版API藍圖...")
    app.register_blueprint(legacy_detection_bp)
    app.register_blueprint(legacy_safety_bp)
    logger.info("舊版API藍圖註冊完成")
    
    # 註冊全局中介軟體
    @app.before_request
    def before_request():
        """請求前處理"""
        # 這裡可以添加全局的請求前處理邏輯
        pass
    
    @app.after_request
    def after_request(response):
        """請求後處理"""
        # 添加安全標頭
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # 添加版本信息
        response.headers['X-API-Version'] = '2.0.0'
        response.headers['X-Service'] = 'FireGuard-AI'
        
        return response
    
    # 錯誤處理
    @app.errorhandler(404)
    def not_found(error):
        """404錯誤處理"""
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested API endpoint does not exist',
                'status_code': 404
            }), 404
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """500錯誤處理"""
        logger.error(f"Internal server error: {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred',
                'status_code': 500
            }), 500
        return render_template('500.html'), 500
    
    @app.errorhandler(429)
    def rate_limit_error(error):
        """429錯誤處理（速率限制）"""
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.',
            'status_code': 429
        }), 429
    
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
    
    # API根路由
    @app.route('/api', methods=['GET'])
    def api_info():
        """API信息端點"""
        return jsonify({
            'service': 'FireGuard AI',
            'version': '2.0.0',
            'description': 'AI-powered fire detection and analysis system',
            'api_versions': {
                'v1': '/api/v1',
                'legacy': '/api'
            },
            'endpoints': {
                'health': '/api/v1/monitoring/health',
                'authentication': '/api/v1/auth',
                'detection': '/api/v1/detection',
                'monitoring': '/api/v1/monitoring'
            },
            'documentation': 'https://docs.fireguard.ai'
        })
    
    # 版本信息路由
    @app.route('/api/version', methods=['GET'])
    def version_info():
        """版本信息"""
        return jsonify({
            'version': '2.0.0',
            'build_date': '2024-01-01',
            'features': [
                'AI fire detection',
                'Real-time analysis', 
                'Multi-user authentication',
                'API rate limiting',
                'Health monitoring',
                'Redis caching',
                'Database persistence'
            ]
        })
    
    # 初始化健康檢查監控
    logger.info("啟動健康檢查監控...")
    try:
        health_checker.start_monitoring()
        logger.info("健康檢查監控已啟動")
    except Exception as e:
        logger.warning(f"健康檢查監控啟動警告: {str(e)}")
    
    # 註冊應用關閉時的清理函數
    def cleanup():
        """應用關閉時的清理工作"""
        logger.info("正在關閉 FireGuard AI 系統...")
        try:
            health_checker.stop_monitoring()
            logger.info("健康檢查監控已停止")
        except Exception as e:
            logger.error(f"清理過程中發生錯誤: {str(e)}")
    
    atexit.register(cleanup)
    
    # 創建數據庫表（開發環境）
    with app.app_context():
        try:
            db.create_all()
            
            # 創建默認管理員用戶（如果不存在）
            from data.repositories.user_repository import UserRepository
            user_repo = UserRepository()
            
            admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
            admin_email = os.environ.get('ADMIN_EMAIL', 'admin@fireguard.ai')
            admin_password = os.environ.get('ADMIN_PASSWORD', 'Admin123!')
            
            if not user_repo.find_by_username(admin_username):
                try:
                    admin_user = user_repo.create_user(
                        username=admin_username,
                        email=admin_email,
                        password=admin_password,
                        role='admin'
                    )
                    logger.info(f"默認管理員用戶已創建: {admin_username}")
                except Exception as e:
                    logger.warning(f"創建管理員用戶失敗: {str(e)}")
            
        except Exception as e:
            logger.error(f"數據庫表創建失敗: {str(e)}")
    
    logger.info("FireGuard AI 系統初始化完成 ✅")
    return app

# 創建應用實例
app = create_app()

if __name__ == '__main__':
    # 開發環境運行
    import os
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger = get_logger(__name__)
    logger.info(f"FireGuard AI 開發服務器啟動於端口 {port}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)