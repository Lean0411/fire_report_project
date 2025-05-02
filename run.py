from app import create_app
from app.utils import setup_logging
from app.config import DevelopmentConfig

# 設定日誌
setup_logging()

# 建立應用實例
app = create_app(DevelopmentConfig)

if __name__ == '__main__':
    app.run(debug=True)