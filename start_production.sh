#!/bin/bash
"""
生產環境啟動腳本
使用 Gunicorn WSGI 伺服器運行 Flask 應用程式
"""

# 啟用虛擬環境
source venv/bin/activate

# 確保日誌目錄存在
mkdir -p logs

# 設定環境變數
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "啟動火災檢測系統 (生產模式)..."
echo "使用 Gunicorn WSGI 伺服器"
echo "配置檔案: gunicorn.conf.py"

# 使用 Gunicorn 啟動應用程式
gunicorn --config gunicorn.conf.py app:app

echo "應用程式已停止"