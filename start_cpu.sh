#!/bin/bash
# 強制使用 CPU 啟動腳本，避免 NVIDIA 驅動問題

echo "🔥 火災檢測系統啟動中 (強制 CPU 模式)..."

# 啟用虛擬環境
source venv/bin/activate

# 設定環境變數強制使用 CPU
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "💻 已設定強制使用 CPU 模式"
echo "🚀 啟動 Flask 應用程式..."

# 啟動應用程式
python app.py