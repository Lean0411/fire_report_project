import os
from datetime import datetime
import logging
import requests
from app.config import Config

# ===== 日誌輔助（可選） =====
def setup_logging():
    """設定應用程式日誌（已在 create_app 中處理，可保留供其他模組呼叫）"""
    os.makedirs(Config.LOG_FOLDER, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

# ===== 檔案處理輔助 =====
def allowed_file(filename: str) -> bool:
    """檢查檔案是否為允許的格式"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def generate_timestamp() -> str:
    """產生時間戳記，用於檔名"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_file_extension(filename: str) -> str:
    """取得檔案副檔名"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

# ===== LLM 呼叫輔助 =====
def call_gemma(prompt: str,
               max_tokens: int = None,
               temperature: float = None) -> str:
    """
    呼叫本地 Ollama-serve 上的 Gemma 模型，回傳文字生成結果。
    max_tokens 與 temperature 預設使用 Config 內的設定。
    """
    url = f"{Config.OLLAMA_HOST}/v1/chat/completions"
    payload = {
        "model": Config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
        # 若未傳入參數，則使用 Config 裡的預設值
        "max_tokens": max_tokens or Config.LLM_MAX_TOKENS,
        "temperature": temperature or Config.LLM_TEMPERATURE
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # 回傳第一個選項的內容
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"呼叫 Gemma 失敗：{e}")
        return "對不起，無法取得模型回應。"
