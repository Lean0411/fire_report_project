"""
Gunicorn 配置檔案
用於生產環境部署的WSGI伺服器配置
"""

import os

# 綁定地址和端口
bind = f"0.0.0.0:{os.environ.get('PORT', '5002')}"

# Worker 設定
workers = int(os.environ.get('WORKERS', '4'))  # 建議為 CPU 核心數 * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# 日誌設定
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 進程設定
preload_app = True
max_requests = 1000
max_requests_jitter = 50

# 安全設定
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# PID 檔案
pidfile = "logs/gunicorn.pid"

# 重新載入
reload = False  # 生產環境設為 False

# SSL 設定 (如果需要)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile"