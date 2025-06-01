#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火災偵測系統自動啟動腳本 (完全自動化版本)
"""

import sys
import os
import subprocess
import socket
import time

# 確保當前目錄在Python路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def check_port(host, port):
    """檢查端口是否被佔用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0  # 0 表示端口被佔用
    except:
        return False

def get_process_using_port(port):
    """獲取佔用指定端口的進程 PID"""
    try:
        # 使用 netstat 查找佔用端口的進程
        cmd = f"netstat -tulpn 2>/dev/null | grep :{port}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTEN' in line:
                    # 提取 PID
                    parts = line.split()
                    for part in parts:
                        if '/' in part:
                            pid = part.split('/')[0]
                            if pid.isdigit():
                                return int(pid)
        return None
    except Exception as e:
        print(f"⚠️ 檢查端口進程時出錯: {e}")
        return None

def kill_process(pid):
    """終止指定的進程"""
    try:
        # 嘗試優雅地終止進程
        subprocess.run(['kill', str(pid)], check=True)
        time.sleep(3)  # 增加等待時間
        
        # 檢查進程是否仍在運行
        try:
            subprocess.run(['kill', '-0', str(pid)], check=True)
            # 如果仍在運行，強制終止
            print(f"🔄 進程 {pid} 仍在運行，強制終止...")
            subprocess.run(['kill', '-9', str(pid)], check=True)
            time.sleep(2)
        except subprocess.CalledProcessError:
            # 進程已經終止
            pass
            
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"⚠️ 終止進程時出錯: {e}")
        return False

def auto_handle_port_conflict(host, port, max_retries=3):
    """自動處理端口衝突，直接終止佔用進程"""
    for retry in range(max_retries):
        print(f"⚠️ 端口 {port} 被佔用，正在自動清理... (嘗試 {retry + 1}/{max_retries})")
        
        pid = get_process_using_port(port)
        if pid:
            print(f"🔍 發現進程 PID: {pid}")
            
            # 獲取進程信息
            try:
                cmd = f"ps -p {pid} -o pid,ppid,cmd --no-headers"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    process_info = result.stdout.strip()
                    print(f"📋 進程信息: {process_info}")
            except:
                pass
            
            print(f"💀 自動終止進程 {pid}...")
            if kill_process(pid):
                print(f"✅ 成功終止進程 {pid}")
                time.sleep(2)  # 等待端口釋放
                
                # 多次檢查端口是否真的釋放
                for check in range(5):
                    if not check_port(host, port):
                        print(f"🎉 端口 {port} 已釋放")
                        return True
                    time.sleep(1)
                    print(f"⏳ 等待端口釋放... ({check + 1}/5)")
                
                print(f"❌ 端口 {port} 仍被佔用，繼續嘗試...")
            else:
                print(f"❌ 無法終止進程 {pid}")
        else:
            print("🤷 無法找到佔用端口的進程，端口可能已釋放")
            if not check_port(host, port):
                return True
            
        if retry < max_retries - 1:
            print(f"🔄 等待 2 秒後重試...")
            time.sleep(2)
    
    return False

print("🔥 火災偵測系統自動啟動中...")

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
    port = int(os.environ.get('PORT', 5002))
    
    print(f"📍 準備啟動於: http://{host}:{port}")
    print(f"🌐 或使用: http://localhost:{port}")
    
    # 檢查端口是否被佔用
    if check_port(host, port):
        print(f"⚠️ 端口 {port} 已被佔用！")
        
        if auto_handle_port_conflict(host, port):
            print("✅ 端口衝突已自動解決，繼續啟動...")
        else:
            print("❌ 無法自動解決端口衝突")
            print(f"💡 建議：")
            print(f"   1. 手動執行: sudo netstat -tulpn | grep :{port}")
            print(f"   2. 或使用其他端口: PORT=5003 python run_auto.py")
            sys.exit(1)
    else:
        print(f"✅ 端口 {port} 可用")
    
    print("=" * 50)
    print("🚀 系統啟動中...")
    
    # 使用app.py中的Flask實例，禁用 reloader 避免雙重啟動
    try:
        app.run(
            host=host,
            port=port,
            debug=True,
            use_reloader=False,  # 禁用自動重載避免進程衝突
            threaded=True
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ 啟動失敗：端口 {port} 仍被佔用")
            print("🔄 請稍等片刻後重新運行腳本")
        else:
            print(f"\n❌ 啟動失敗：{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n👋 用戶中斷，系統已關閉")
        sys.exit(0) 