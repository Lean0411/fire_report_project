"""
Gunicorn 配置檔案
用於生產環境部署的WSGI伺服器配置
"""

import os
from config.constants import (
    DEFAULT_WORKERS, WORKER_CONNECTIONS, WORKER_TIMEOUT,
    WORKER_KEEPALIVE, MAX_REQUESTS, MAX_REQUESTS_JITTER,
    LIMIT_REQUEST_LINE, LIMIT_REQUEST_FIELDS, LIMIT_REQUEST_FIELD_SIZE,
    DEFAULT_PORT
)

# 綁定地址和端口
bind = f"0.0.0.0:{os.environ.get('PORT', str(DEFAULT_PORT))}"

# Worker 設定
workers = int(os.environ.get('WORKERS', str(DEFAULT_WORKERS)))  # 建議為 CPU 核心數 * 2 + 1
worker_class = "sync"
worker_connections = WORKER_CONNECTIONS
timeout = WORKER_TIMEOUT
keepalive = WORKER_KEEPALIVE

# 日誌設定
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 進程設定
preload_app = True
max_requests = MAX_REQUESTS
max_requests_jitter = MAX_REQUESTS_JITTER

# 安全設定
limit_request_line = LIMIT_REQUEST_LINE
limit_request_fields = LIMIT_REQUEST_FIELDS
limit_request_field_size = LIMIT_REQUEST_FIELD_SIZE

# PID 檔案
pidfile = "logs/gunicorn.pid"

# 重新載入
reload = False  # 生產環境設為 False

# SSL 設定 (如果需要)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile"