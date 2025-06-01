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
import openai
import requests
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

app = Flask(__name__)

# ===================== OpenAI 設定 =====================
openai.api_key = os.getenv('OPENAI_API_KEY')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma:7b')

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

def filter_refusal_responses(text: str) -> str:
    """過濾掉AI的拒絕回應，只保留有用的內容"""
    if not text or not text.strip():
        return ""
    
    # 定義拒絕回應的關鍵詞和模式
    refusal_patterns = [
        "抱歉，我無法識別或描述圖片中的具體內容",
        "我無法識別或描述圖片中的具體內容",
        "很抱歉，我無法識別或描述圖片中的人物或具體細節",
        "我無法識別或描述圖片中的人物或具體細節",
        "抱歉，我無法識別圖片",
        "我無法識別圖片",
        "很抱歉，我無法識別圖片",
        "抱歉，我無法分析圖片",
        "我無法分析圖片",
        "很抱歉，我無法分析圖片",
        "我無法查看或分析圖片",
        "抱歉，我無法查看或分析圖片",
        "很抱歉，我無法查看或分析圖片",
        "我無法直接識別圖片內容",
        "抱歉，我無法直接識別圖片內容",
        "很抱歉，我無法直接識別圖片內容",
        "我無法看到圖片",
        "抱歉，我無法看到圖片",
        "很抱歉，我無法看到圖片",
        "我無法識別圖片中的人物",
        "抱歉，我無法識別圖片中的人物",
        "很抱歉，我無法識別圖片中的人物",
        "我無法描述圖片中的具體細節",
        "抱歉，我無法描述圖片中的具體細節",
        "很抱歉，我無法描述圖片中的具體細節"
    ]
    
    # 分割文本為段落
    paragraphs = text.split('\n')
    filtered_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # 檢查是否包含拒絕模式
        contains_refusal = False
        for pattern in refusal_patterns:
            if pattern in paragraph:
                contains_refusal = True
                break
        
        # 如果不包含拒絕模式，保留這個段落
        if not contains_refusal:
            filtered_paragraphs.append(paragraph)
    
    filtered_text = '\n'.join(filtered_paragraphs)
    
    # 如果過濾後的文本太短或只包含一般性建議，返回空字串
    if len(filtered_text.strip()) < 50:
        app.logger.warning("過濾後的AI回應內容太短，可能是無效回應")
        return ""
    
    # 檢查是否只包含一般性的無意義內容
    generic_phrases = [
        "不過，我可以提供一些一般的消防安全建議",
        "以下是一些一般的消防安全建議",
        "我可以提供一些基本的消防安全建議",
        "以下是一些基本的消防安全建議",
        "不過，我可以根據火災情境提供一般的消防安全建議",
        "我可以根據火災情境提供一般的消防安全建議",
        "不過，我可以提供一般性的消防安全建議",
        "我可以提供一般性的消防安全建議"
    ]
    
    # 如果包含這些一般性短語，且沒有其他具體內容，過濾掉
    for phrase in generic_phrases:
        if phrase in filtered_text and len(filtered_text.replace(phrase, "").strip()) < 100:
            app.logger.warning("檢測到一般性無意義回應，已過濾")
            return ""
    
    app.logger.info(f"AI回應過濾完成，原始長度：{len(text)}，過濾後長度：{len(filtered_text)}")
    return filtered_text

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
    
    # 從表單獲取設定
    use_ai = request.form.get('use_ai', 'false').lower() == 'true'
    ai_provider = request.form.get('ai_provider', 'openai')
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
    
    # 基礎 AI 報告
    llm_report = ""
    
    if use_ai and is_fire:
        # 構建提示詞
        role_map = {
            'general': '一般民眾',
            'firefighter': '消防隊員', 
            'management': '管理單位'
        }
        prompt = f"""作為專業消防安全專家，您必須分析提供的圖片內容。

檢測資訊：
- CNN 模型結果：火災機率 {p_fire:.1%}
- 目標用戶：{role_map.get(role, '一般民眾')}

請觀察圖片並描述：
1. **場景內容**：描述圖片中可見的物件、環境、顏色和佈局
2. **安全狀況**：基於所見內容評估潛在的火災風險
3. **安全建議**：針對{role_map.get(role, '一般民眾')}提供具體的預防措施
4. **應急準備**：提供火災發生時的應對方案

即使圖片內容簡單，也請基於可見元素提供實用的安全建議。請用繁體中文回答。"""
        
        if ai_provider == 'openai':
            llm_report = call_openai_gpt(prompt, image_path=save_path)
        elif ai_provider == 'ollama':
            llm_report = call_ollama_gemma(prompt)
        else:
            llm_report = "未知的 AI 提供者"
    
    # 加入基礎火災安全建議
    fire_safety_tips = generate_fire_safety_tips() if is_fire else {}
    
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
            'recommendations': recommendations,
            'llm_report': llm_report,
            'fire_safety_tips': fire_safety_tips,
            'ai_provider_used': ai_provider if use_ai and llm_report else None
        }
    }
    
    return jsonify(result)

@app.route('/api/fire-safety-advice', methods=['GET', 'POST'])
def get_fire_safety_advice():
    """提供火災安全建議的專用 API"""
    try:
        if request.method == 'GET':
            scenario = request.args.get('scenario', 'general')
            use_ai = request.args.get('use_ai', 'false').lower() == 'true'
            ai_provider = request.args.get('ai_provider', 'openai')
        else:  # POST
            data = request.get_json() if request.is_json else {}
            scenario = data.get('scenario', 'general')
            use_ai = data.get('use_ai', False)
            ai_provider = data.get('ai_provider', 'openai')
        
        # 基礎安全建議
        safety_tips = generate_fire_safety_tips()
        
        # AI 生成的個性化建議
        ai_advice = ""
        if use_ai:
            scenario_prompts = {
                'home': '請針對家庭火災提供詳細的預防措施和應急處置建議',
                'office': '請針對辦公室火災提供詳細的預防措施和應急處置建議',
                'factory': '請針對工廠火災提供詳細的預防措施和應急處置建議',
                'kitchen': '請針對廚房火災提供詳細的預防措施和應急處置建議',
                'electrical': '請針對電器火災提供詳細的預防措施和應急處置建議',
                'general': '請提供一般性的火災預防和應急處置建議'
            }
            
            prompt = scenario_prompts.get(scenario, scenario_prompts['general'])
            prompt += "\n\n請用繁體中文回答，並分為「預防措施」、「應急處置」、「注意事項」三個部分。"
            
            if ai_provider == 'openai':
                ai_advice = call_openai_gpt(prompt)
            elif ai_provider == 'ollama':
                ai_advice = call_ollama_gemma(prompt)
        
        result = {
            'success': True,
            'data': {
                'scenario': scenario,
                'fire_safety_tips': safety_tips,
                'ai_advice': ai_advice,
                'ai_provider_used': ai_provider if use_ai and ai_advice else None
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"取得火災安全建議失敗: {e}")
        return jsonify({
            'success': False, 
            'error': f'取得火災安全建議失敗: {str(e)}'
        }), 500

def generate_fire_safety_tips():
    """生成火災安全建議"""
    return {
        "immediate_actions": [
            "立即撥打119報警，說明火災位置和規模",
            "如在室內，請低身爬行避免濃煙，用濕毛巾掩住口鼻",
            "觸摸門把前先檢查溫度，如燙手請勿開門",
            "若火勢小且安全，可使用滅火器撲滅，記住「拉、瞄、壓、掃」四步驟",
            "確保逃生路線暢通，不可使用電梯"
        ],
        "evacuation_tips": [
            "選擇最近的安全出口，避開濃煙區域",
            "幫助行動不便者一同撤離",
            "到達安全地點後立即清點人數",
            "不要返回火場搶救物品",
            "在安全地點等待消防人員到達"
        ],
        "prevention_measures": [
            "定期檢查電線和電器設備，避免過載使用",
            "保持逃生通道暢通，不堆放雜物",
            "家中準備滅火器、煙霧偵測器和逃生繩",
            "制定家庭火災逃生計畫並定期演練",
            "廚房用火後務必檢查爐火是否完全熄滅"
        ],
        "high_rise_building": [
            "熟悉大樓緊急逃生路線和集合點",
            "火災時絕對不可使用電梯",
            "若煙霧瀰漫樓梯間，考慮就地避難等待救援",
            "在安全樓層等待消防人員指示",
            "使用濕毛巾密封門縫防止煙霧進入"
        ]
    }

def call_openai_gpt(prompt: str, model: str = "gpt-4o", image_path: str = None) -> str:
    """呼叫 OpenAI GPT 模型生成火災應急建議，支援圖片分析"""
    try:
        if not openai.api_key:
            return "未設置 OpenAI API Key，請在 .env 檔案中設置 OPENAI_API_KEY"
        
        messages = [
            {"role": "system", "content": "你是一個專業的消防安全專家，能夠分析圖片並提供實用的火災應急建議。請用繁體中文回答，不要回答「抱歉，我無法識別或描述這張圖片」這種資訊"}
        ]
        
        # 如果有圖片，使用 Vision API
        if image_path and os.path.exists(image_path):
            import base64
            from PIL import Image as PILImage
            
            # 確保圖片格式正確並重新保存為 JPEG
            try:
                # 開啟並轉換圖片
                img = PILImage.open(image_path).convert('RGB')
                
                # 調整圖片大小以符合 OpenAI 要求 (最大 2048x2048)
                max_size = 2048
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
                
                # 保存為臨時 JPEG 檔案
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    img.save(temp_file.name, 'JPEG', quality=85)
                    temp_path = temp_file.name
                
                # 讀取並編碼圖片
                with open(temp_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # 清理臨時檔案
                os.unlink(temp_path)
                
                # 檢測實際圖片格式
                mime_type = "image/jpeg"
                
                app.logger.info(f"成功編碼圖片，base64 長度：{len(base64_image)} 字元")
                
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
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"  # 使用高解析度分析
                            }
                        }
                    ]
                })
                
            except Exception as img_error:
                app.logger.error(f"圖片處理失敗: {img_error}")
                # 如果圖片處理失敗，退回到純文字模式
                messages.append({"role": "user", "content": f"{prompt}\n\n注意：圖片處理失敗，請基於 CNN 檢測結果提供分析。"})
        else:
            messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1500,  # 增加 token 數量
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        app.logger.info(f"OpenAI API 成功回應，內容長度：{len(result)} 字元")
        
        # 過濾掉無用的拒絕回應
        result = filter_refusal_responses(result)
        
        return result
        
    except Exception as e:
        app.logger.error(f"OpenAI API 呼叫失敗: {e}")
        error_msg = str(e)
        
        # 提供更詳細的錯誤訊息
        if "api_key" in error_msg.lower():
            return "OpenAI API Key 無效或未設置，請檢查 .env 檔案中的 OPENAI_API_KEY"
        elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
            return "OpenAI API 額度不足，請檢查您的帳戶餘額"
        elif "rate_limit" in error_msg.lower():
            return "OpenAI API 請求頻率過高，請稍後再試"
        else:
            return f"OpenAI API 呼叫失敗: {error_msg}"

def call_ollama_gemma(prompt: str) -> str:
    """呼叫本地 Ollama Gemma 模型"""
    try:
        url = f"{OLLAMA_HOST}/v1/chat/completions"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "你是一個專業的消防安全專家，請提供實用的火災應急建議。請用繁體中文回答。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.3
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        app.logger.error(f"Ollama 呼叫失敗: {e}")
        return f"本地 LLM 暫時無法使用: {str(e)}"

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5002
    print(f"==> 請在瀏覽器中訪問: http://127.0.0.1:{port}")
    print(f"==> 或使用: http://localhost:{port}")
    app.run(host=host, port=port, debug=True)
