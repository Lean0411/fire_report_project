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
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    debug = app.config.get('DEBUG', False)
    print(f"==> Starting server at http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
