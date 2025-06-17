"""
模型工具模組
提供模型載入、預測和管理相關功能
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
from typing import Optional, Tuple
import os
import threading
from .cnn_model import DeepCNN
from config.settings import Config
from config.logging_config import get_logger

logger = get_logger(__name__)

class ModelManager:
    """模型管理器，負責模型的載入、預測和管理"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[DeepCNN] = None
        self._model_lock = threading.Lock()
        self._load_attempted = False
        
        # 圖像預處理管道
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"模型管理器初始化完成，使用設備: {self.device}")
    
    def load_model(self) -> bool:
        """
        載入CNN模型權重
        
        Returns:
            bool: 載入是否成功
        """
        with self._model_lock:
            # 如果模型已載入，直接返回
            if self.model is not None:
                return True
            
            # 如果已嘗試載入且失敗，避免重複嘗試
            if self._load_attempted:
                return False
            
            try:
                logger.info("開始載入火災檢測模型...")
                self._load_attempted = True
                
                # 檢查模型檔案是否存在
                model_path = Config.MODEL_PATH
                if not os.path.exists(model_path):
                    logger.error(f"模型檔案不存在: {model_path}")
                    return False
                
                # 建立模型實例
                self.model = DeepCNN().to(self.device)
                
                # 載入權重
                state_dict = torch.load(
                    model_path, 
                    map_location=self.device, 
                    weights_only=True
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                
                # 記錄模型資訊
                model_info = self.model.get_model_info()
                logger.info(f"模型載入成功: {model_info}")
                
                return True
                
            except Exception as e:
                logger.error(f"模型載入失敗: {e}")
                self.model = None
                return False
    
    def predict_fire(self, image_path: str) -> Optional[Tuple[float, float]]:
        """
        對圖像進行火災檢測預測
        
        Args:
            image_path: 圖像檔案路徑
            
        Returns:
            Optional[Tuple[float, float]]: (火災機率, 安全機率) 或 None
        """
        # 確保模型已載入
        if not self.load_model():
            logger.error("模型未載入，無法進行預測")
            return None
        
        try:
            # 檢查檔案是否存在
            if not os.path.exists(image_path):
                logger.error(f"圖像檔案不存在: {image_path}")
                return None
            
            # 載入和預處理圖像
            image = PILImage.open(image_path).convert('RGB')
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 進行預測
            with torch.no_grad():
                logit = self.model(tensor).item()
                
                # sigmoid(logit) = P(NoFire)
                p_no_fire = torch.sigmoid(torch.tensor(logit)).item()
                p_fire = 1.0 - p_no_fire
            
            logger.info(f"預測完成 - 火災機率: {p_fire:.3f}, 安全機率: {p_no_fire:.3f}")
            return p_fire, p_no_fire
            
        except Exception as e:
            logger.error(f"預測過程發生錯誤: {e}")
            return None
    
    def get_device_info(self) -> dict:
        """
        獲取設備資訊
        
        Returns:
            dict: 設備資訊
        """
        info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_current_device': torch.cuda.current_device(),
                'cuda_device_name': torch.cuda.get_device_name()
            })
        
        return info
    
    def get_model_status(self) -> dict:
        """
        獲取模型狀態
        
        Returns:
            dict: 模型狀態資訊
        """
        return {
            'model_loaded': self.model is not None,
            'load_attempted': self._load_attempted,
            'device': str(self.device),
            'model_path': Config.MODEL_PATH,
            'model_exists': os.path.exists(Config.MODEL_PATH)
        }
    
    def unload_model(self):
        """卸載模型以釋放記憶體"""
        with self._model_lock:
            if self.model is not None:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("模型已卸載")

# 全域模型管理器實例
model_manager = ModelManager()

# 為了保持向後兼容性的函數
def load_model() -> bool:
    """載入模型（向後兼容）"""
    return model_manager.load_model()

def predict_fire(image_path: str) -> Optional[Tuple[float, float]]:
    """預測火災（向後兼容）"""
    return model_manager.predict_fire(image_path)