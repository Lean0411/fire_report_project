"""
CNN模型定義模組
包含深度學習模型架構定義
"""
import torch
import torch.nn as nn
from config.constants import (
    CNN_CHANNELS_INPUT, CNN_CHANNELS_L1, CNN_CHANNELS_L2,
    CNN_CHANNELS_L3, CNN_CHANNELS_L4, CNN_KERNEL_SIZE,
    CNN_POOL_SIZE, CNN_HIDDEN_SIZE, CNN_DROPOUT_RATE,
    CNN_FEATURE_MAP_SIZE, IMAGE_INPUT_SIZE
)

class DeepCNN(nn.Module):
    """
    深度卷積神經網路模型，用於火災檢測
    
    架構：
    - 4個卷積區塊，每個區塊包含卷積層、批次正規化、ReLU激活和最大池化
    - 分類器：全連接層進行二元分類（火災/非火災）
    """
    
    def __init__(self):
        super(DeepCNN, self).__init__()
        
        # 特徵提取層
        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(CNN_CHANNELS_INPUT, CNN_CHANNELS_L1, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS_L1, CNN_CHANNELS_L1, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(CNN_POOL_SIZE),  # 224->112
            
            # Block 2: 64 -> 128
            nn.Conv2d(CNN_CHANNELS_L1, CNN_CHANNELS_L2, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS_L2, CNN_CHANNELS_L2, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(CNN_POOL_SIZE),  # 112->56
            
            # Block 3: 128 -> 256
            nn.Conv2d(CNN_CHANNELS_L2, CNN_CHANNELS_L3, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L3), 
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS_L3, CNN_CHANNELS_L3, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L3), 
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS_L3, CNN_CHANNELS_L3, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L3), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(CNN_POOL_SIZE),  # 56->28
            
            # Block 4: 256 -> 512
            nn.Conv2d(CNN_CHANNELS_L3, CNN_CHANNELS_L4, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L4), 
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS_L4, CNN_CHANNELS_L4, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L4), 
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS_L4, CNN_CHANNELS_L4, CNN_KERNEL_SIZE, padding=1), 
            nn.BatchNorm2d(CNN_CHANNELS_L4), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(CNN_POOL_SIZE)   # 28->14
        )
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(CNN_CHANNELS_L4 * CNN_FEATURE_MAP_SIZE * CNN_FEATURE_MAP_SIZE, CNN_HIDDEN_SIZE), 
            nn.ReLU(inplace=True), 
            nn.Dropout(CNN_DROPOUT_RATE),
            nn.Linear(CNN_HIDDEN_SIZE, 1)  # 單一 logit，代表 NoFire 類別
        )

    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入張量 (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: 輸出 logit (batch_size,)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平為一維
        return self.classifier(x).squeeze(1)

    def get_model_info(self) -> dict:
        """
        獲取模型資訊
        
        Returns:
            dict: 模型資訊字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'DeepCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (CNN_CHANNELS_INPUT,) + IMAGE_INPUT_SIZE,
            'output_size': 1,
            'task': 'Binary Classification (Fire/No Fire)'
        }