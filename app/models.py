import torch
import torch.nn as nn
from torchvision import transforms
import logging
from app.config import Config

logger = logging.getLogger(__name__)

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224->112
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112->56
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56->28
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 28->14
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 1)  # 單一 logit，代表 NoFire 類別
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(1)

class FireDetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocess = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """載入模型權重"""
        if self.model is not None:
            return True
        try:
            self.model = DeepCNN().to(self.device)
            state = torch.load(Config.MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info("模型載入成功")
            return True
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            return False

    def predict(self, image):
        """對輸入圖片進行預測"""
        if not self.load_model():
            return None, None
        
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor).item()
            p_no = torch.sigmoid(torch.tensor(logit)).item()
            p_fire = 1.0 - p_no
        return p_fire, p_no

# 全域模型實例
model = FireDetectionModel()