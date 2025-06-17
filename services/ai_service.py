"""
AI服務模組
提供與外部AI模型（OpenAI、Ollama）的整合功能
"""
import os
import base64
import tempfile
import requests
from typing import Optional
from PIL import Image as PILImage
import openai

from config.settings import Config
from config.logging_config import get_logger
from utils.text_utils import filter_refusal_responses

logger = get_logger(__name__)

class AIService:
    """AI服務類，負責與AI模型的溝通"""
    
    def __init__(self):
        self.openai_config = Config.get_openai_config()
        self.ollama_config = Config.get_ollama_config()
        
        # 設置 OpenAI API Key
        if self.openai_config['api_key']:
            openai.api_key = self.openai_config['api_key']
    
    def call_openai_gpt(self, prompt: str, model: str = "gpt-4o", 
                       image_path: Optional[str] = None) -> str:
        """
        呼叫 OpenAI GPT 模型生成火災應急建議，支援圖片分析
        
        Args:
            prompt: 提示文字
            model: 使用的模型名稱
            image_path: 圖片路徑（可選）
            
        Returns:
            str: AI 生成的回應
        """
        try:
            if not openai.api_key:
                return "未設置 OpenAI API Key，請在 .env 檔案中設置 OPENAI_API_KEY"
            
            messages = [
                {
                    "role": "system", 
                    "content": "你是一個專業的消防安全專家，能夠分析圖片並提供實用的火災應急建議。請用繁體中文回答，不要回答「抱歉，我無法識別或描述這張圖片」這種資訊"
                }
            ]
            
            # 如果有圖片，使用 Vision API
            if image_path and os.path.exists(image_path):
                base64_image = self._encode_image(image_path)
                if base64_image:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    })
                else:
                    # 如果圖片編碼失敗，僅使用文字
                    messages.append({"role": "user", "content": prompt})
            else:
                messages.append({"role": "user", "content": prompt})
            
            # 呼叫 OpenAI API
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # 過濾拒絕回應
            filtered_content = filter_refusal_responses(content)
            if not filtered_content:
                logger.warning("OpenAI 回應被過濾，可能包含拒絕內容")
                return self._generate_fallback_response()
            
            logger.info(f"OpenAI 成功回應，內容長度：{len(filtered_content)} 字元")
            return filtered_content
            
        except Exception as e:
            logger.error(f"OpenAI API 呼叫失敗: {e}")
            return self._generate_fallback_response()
    
    def call_ollama_gemma(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        呼叫本地 Ollama Gemma 模型
        
        Args:
            prompt: 提示文字
            image_path: 圖片路徑（目前不支援）
            
        Returns:
            str: AI 生成的回應
        """
        try:
            url = f"{self.ollama_config['host']}/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.ollama_config['model'],
                "messages": [
                    {
                        "role": "system",
                        "content": "你是專業的消防安全專家，請提供實用的火災應急建議。用繁體中文回答。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 過濾拒絕回應
            filtered_content = filter_refusal_responses(content)
            if not filtered_content:
                logger.warning("Ollama 回應被過濾，可能包含拒絕內容")
                return self._generate_fallback_response()
            
            logger.info(f"Ollama 成功回應，內容長度：{len(filtered_content)} 字元")
            return filtered_content
            
        except requests.exceptions.ConnectionError:
            logger.error("無法連接到 Ollama 服務，請確認服務是否運行")
            return self._generate_fallback_response()
        except Exception as e:
            logger.error(f"Ollama 呼叫失敗: {e}")
            return self._generate_fallback_response()
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """
        編碼圖片為 base64 格式
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            Optional[str]: base64 編碼的圖片或 None
        """
        try:
            # 開啟並轉換圖片
            img = PILImage.open(image_path).convert('RGB')
            
            # 調整圖片大小以符合 OpenAI 要求 (最大 2048x2048)
            max_size = 2048
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
            
            # 保存為臨時 JPEG 檔案
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                img.save(temp_file.name, 'JPEG', quality=85)
                temp_path = temp_file.name
            
            # 讀取並編碼圖片
            with open(temp_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 清理臨時檔案
            os.unlink(temp_path)
            
            logger.info(f"圖片編碼成功，base64 長度：{len(base64_image)} 字元")
            return base64_image
            
        except Exception as e:
            logger.error(f"圖片編碼失敗: {e}")
            return None
    
    def _generate_fallback_response(self) -> str:
        """
        生成備用回應
        
        Returns:
            str: 備用回應文字
        """
        return """基於火災檢測結果，建議採取以下應急措施：

1. **立即評估現場安全**
   - 確認火源位置和火勢大小
   - 評估疏散路線是否安全
   - 檢查是否有人員受困

2. **啟動應急程序**
   - 立即撥打119報警
   - 啟動火災警報系統
   - 組織人員有序疏散

3. **初期滅火行動**
   - 如火勢較小，可使用適當滅火器
   - 切斷電源和關閉燃氣閥門
   - 避免使用水撲滅電器火災

4. **人員安全措施**
   - 優先確保人員生命安全
   - 協助行動不便者疏散
   - 在安全地點集合清點人數

請根據現場實際情況靈活應對，生命安全永遠是第一優先。"""

    def get_service_status(self) -> dict:
        """
        獲取AI服務狀態
        
        Returns:
            dict: 服務狀態資訊
        """
        return {
            'openai_configured': bool(self.openai_config['api_key']),
            'ollama_host': self.ollama_config['host'],
            'ollama_model': self.ollama_config['model']
        }

# 全域 AI 服務實例
ai_service = AIService()