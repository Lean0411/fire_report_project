import json
from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import transforms
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import traceback
from datetime import datetime
import sys

app = Flask(__name__)

# ===================== 日誌設定 =====================
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 設定主要日誌處理器
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    
    # 檔案處理器 (每個檔案最大 10MB，保留最後 5 個檔案)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    # 設定 Flask logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)
    
    # 設定 Werkzeug logger
    logging.getLogger('werkzeug').addHandler(file_handler)

setup_logging()

# ===================== 基本設定 =====================
app.config.update(
    UPLOAD_FOLDER='static/uploads/',
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'},
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5MB 上限
    SEND_FILE_MAX_AGE_DEFAULT=0  # 開發模式下停用快取
)

# 確保必要的目錄存在
for directory in [app.config['UPLOAD_FOLDER'], 'models/fire_detection', 'static']:
    os.makedirs(directory, exist_ok=True)

def load_sop_recommendations():
    """載入 SOP 建議內容"""
    try:
        with open('knowledge_base/sop.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        app.logger.error(f"無法載入 SOP 知識庫: {e}")
        return {}

def get_role_recommendations(role, is_fire):
    """根據角色和火災狀況取得建議"""
    sop_data = load_sop_recommendations()
    if not sop_data or role not in sop_data:
        return {}
    
    role_sop = sop_data[role]
    recommendations = {}
    
    if is_fire:
        # 火災情況下，提供所有相關建議
        recommendations = role_sop
    else:
        # 非火災情況下，只提供預防性建議
        if role == "general":
            recommendations["emergency_action_plan"] = role_sop["emergency_action_plan"]
        elif role == "firefighter":
            recommendations["initial_assessment"] = role_sop["initial_assessment"]
        elif role == "management":
            recommendations["emergency_management_protocols"] = role_sop["emergency_management_protocols"]
    
    return recommendations

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/apple-touch-icon.png')
@app.route('/apple-touch-icon-precomposed.png')
def apple_touch_icon():
    return app.send_static_file('apple-touch-icon.png')

# ===================== 二元分類模型定義 =====================
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

# ===================== 全域設定 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 類別標籤對應表（用於前端顯示）
category_labels = {
    'general': '一般民眾',
    'firefighter': '消防隊員', 
    'management': '管理單位',
    'emergency_action_plan': '緊急行動計劃',
    'evacuation_procedures': '疏散程序',
    'evacuation_preparedness': '疏散準備',
    'shelter_in_place': '就地避難',
    'communication_protocol': '通訊協議',
    'initial_assessment': '初步評估',
    'suppression_strategy': '滅火策略',
    'safety_protocols': '安全協議',
    'emergency_management_protocols': '緊急管理協議',
    'resource_allocation': '資源配置',
    'public_communication': '公眾溝通',
    'incident_command': '事故指揮',
    'tactical_operations': '戰術行動',
    'personnel_safety': '人員安全',
    'communication_coordination': '溝通協調',
    'resilience_and_recovery': '復原與重建'
}

# ===================== 輔助函式 =====================
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model() -> bool:
    """載入二元分類模型權重"""
    global model
    if model is not None:
        return True
    try:
        model = DeepCNN().to(device)
        ckpt = 'models/fire_detection/deep_wildfire_cnn_model_amp.pth'
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return True
    except Exception as e:
        app.logger.error(f"模型載入失敗: {e}")
        return False


def predict_fire(path: str):
    """對影像進行 Fire vs NoFire 預測"""
    if not load_model():
        return None, None
    img = PILImage.open(path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor).item()
        # sigmoid(logit) = P(NoFire)
        p_no = torch.sigmoid(torch.tensor(logit)).item()
        p_fire = 1.0 - p_no
    return p_fire, p_no


def generate_annotated_image(image_path: str, filename: str, is_fire: bool, p_fire: float, p_no: float) -> str:
    """生成分析結果圖片（不添加任何標註，保持原圖）"""
    try:
        # 直接複製原圖，不添加任何標註
        from PIL import Image
        
        # 開啟原圖
        img = Image.open(image_path).convert('RGB')
        
        # 儲存圖片（不做任何修改）
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        img.save(annotated_path, quality=95)
        
        app.logger.info(f"成功生成分析圖片（無標註）: {annotated_filename}")
        return annotated_filename
        
    except Exception as e:
        app.logger.error(f"生成分析圖片時發生錯誤: {e}")
        return filename  # 如果失敗，返回原檔名

# ===================== 路由 =====================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return detect_fire()
    return render_template('index.html', category_labels=category_labels)

@app.route('/api/detect', methods=['POST'])
def detect_fire():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未收到檔案'}), 400
    
    role = request.form.get('role', 'general')
    if role not in ['general', 'firefighter', 'management']:
        return jsonify({'success': False, 'error': '無效的身份類型'}), 400
        
    f = request.files['file']
    if f.filename == '':
        return jsonify({'success': False, 'error': '未選擇檔案'}), 400
    if not allowed_file(f.filename):
        return jsonify({'success': False, 'error': '檔案格式不支援'}), 400

    # 生成時間戳檔名以避免衝突
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_ext = os.path.splitext(f.filename)[1]
    fn = f"{timestamp}{file_ext}"
    
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    f.save(save_path)

    p_fire, p_no = predict_fire(save_path)
    if p_fire is None:
        return jsonify({'success': False, 'error': '模型載入失敗'}), 500
    
    is_fire = p_fire > 0.5
    
    # 生成分析結果圖片
    try:
        annotated_filename = generate_annotated_image(save_path, fn, is_fire, p_fire, p_no)
    except Exception as e:
        app.logger.error(f"生成分析圖片失敗: {e}")
        annotated_filename = fn  # 如果生成失敗，使用原圖
    
    # 根據角色獲取建議
    recommendations = get_role_recommendations(role, is_fire)
    
    result = {
        'success': True,
        'data': {
            'filename': annotated_filename,  # 返回標註後的檔名
            'detection': {
                'is_fire': is_fire,
                'description': f'偵測到火災 (信心度: {p_fire:.1%})' if is_fire else f'未偵測到火災 (信心度: {p_no:.1%})',
                'fire_probability': round(p_fire * 100, 1),
                'no_fire_probability': round(p_no * 100, 1)
            },
            'recommendations': recommendations
        }
    }
    
    return jsonify(result)

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5001
    print(f"==> 請在瀏覽器中訪問: http://127.0.0.1:{port}")
    print(f"==> 或使用: http://localhost:{port}")
    app.run(host=host, port=port, debug=True)
