import json
from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import transforms
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

# ===================== 路由 =====================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return detect_fire()
    return render_template('index.html')

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

    fn = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    f.save(save_path)

    p_fire, p_no = predict_fire(save_path)
    if p_fire is None:
        return jsonify({'success': False, 'error': '模型載入失敗'}), 500
    
    is_fire = p_fire > 0.5
    
    # 根據角色獲取建議
    recommendations = get_role_recommendations(role, is_fire)
    
    result = {
        'success': True,
        'data': {
            'filename': fn,
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
    app.run(debug=True)
