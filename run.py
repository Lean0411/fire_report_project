print(">>> ENTERING run.py <<<")

from app import create_app
from app.config import get_config
import os

# run.py

from app import create_app
from app.config import get_config
import os

# 選擇開發或生產設定
config_class = get_config()

# 建立 Flask 應用
app = create_app(config_class)

# 確認程式有載入
print(f"==> run.py loaded; DEBUG={app.config['DEBUG']}")

if __name__ == '__main__':
    host = '127.0.0.1'
    port = int(os.environ.get('PORT', 5001))
    debug = app.config.get('DEBUG', False)
    
    print(f"==> Starting server at http://{host}:{port} (debug={debug})")
    print(f"==> 請在瀏覽器中訪問: http://127.0.0.1:{port}")
    print(f"==> 或使用: http://localhost:{port}")
    
    # 明確設定所有參數，避免被覆蓋
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # 避免重載器可能造成的配置問題
    )
