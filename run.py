#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火災偵測系統啟動腳本
"""

import sys
import os

# 確保當前目錄在Python路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🔥 火災偵測系統啟動中...")

try:
    # 直接導入app.py中的Flask實例
    from app import app
    print("✅ 系統載入成功")
    
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    print("請確認依賴已正確安裝")
    sys.exit(1)

if __name__ == '__main__':
    host = '127.0.0.1'
    port = int(os.environ.get('PORT', 5001))
    
    print(f"📍 網址: http://{host}:{port}")
    print(f"🌐 或使用: http://localhost:{port}")
    print("=" * 40)
    
    # 使用app.py中的Flask實例
    app.run(
        host=host,
        port=port,
        debug=True,
        threaded=True
    )
