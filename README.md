# 🔥 火災偵測系統

![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square)

基於深度學習的智能火災偵測與應急處置系統

[🚀 快速開始](#快速開始) • [📖 使用說明](#使用說明) • [🔧 API文檔](#api-端點) • [❓ 問題回報](https://github.com/Lean0411/fire_report_project/issues)

## 目錄

- [主要功能](#主要功能)
- [系統架構](#系統架構)
  - [核心技術](#核心技術)
  - [檔案結構](#檔案結構)
- [快速開始](#快速開始)
  - [1. 環境準備](#1-環境準備)
  - [2. 模型設定](#2-模型設定)
  - [3. 啟動系統](#3-啟動系統)
- [使用說明](#使用說明)
  - [基本使用流程](#基本使用流程)
  - [API 端點](#api-端點)
- [使用範例](#使用範例)
  - [基本 Web 界面使用](#1-基本-web-界面使用)
  - [API 程式化調用](#2-api-程式化調用)
- [API 文檔](#api-文檔)
  - [POST /api/detect](#post-apidetect---火災檢測)
  - [GET /api/fire-safety-advice](#get-apifire-safety-advice---火災安全建議)
  - [GET /](#get----web-界面)
- [設定說明](#設定說明)
  - [環境變數](#環境變數-env)
  - [角色建議系統](#角色建議系統)
- [技術特色](#技術特色)
- [系統需求](#系統需求)
  - [硬體需求](#硬體需求)
  - [軟體需求](#軟體需求)
- [故障排除](#故障排除)
  - [常見問題](#常見問題)
  - [日誌查看](#日誌查看)
- [貢獻指南](#貢獻指南)
- [授權條款](#授權條款)

## 主要功能

- **智能火災偵測**：使用 CNN 深度學習模型進行火災識別
- **智能分析報告**：整合本地大語言模型提供專業建議
- **角色化建議**：根據使用者身份（一般民眾/消防隊員/管理單位）提供個性化建議
- **視覺化界面**：直觀的 Web 界面，支援拖拽上傳
- **專業 SOP**：內建火災應急處置標準作業程序

## 系統架構

### 核心技術
- **深度學習引擎**：自訓練的 CNN 模型，專門針對火災場景優化
- **本地語言引擎**：支援 Ollama/Gemma 等本地大語言模型
- **Flask 後端**：RESTful API 設計，支援圖片上傳與分析
- **響應式前端**：現代化 Web 界面，支援多種設備

### 檔案結構
```
fire_report_project/
├── app.py                 # Flask 主應用程式
├── run.py                 # 系統啟動腳本
├── requirements.txt       # 依賴套件清單
├── api/                   # API 模組
│   ├── detection.py      # 火災檢測 API
│   └── safety.py         # 安全建議 API
├── config/                # 配置模組
│   ├── settings.py       # 應用配置
│   └── logging_config.py # 日誌配置
├── services/              # 業務邏輯層
│   ├── ai_service.py     # AI 服務
│   ├── image_service.py  # 圖片處理服務
│   ├── sop_service.py    # SOP 服務
│   └── safety_service.py # 安全服務
├── models/                # 模型目錄
│   ├── cnn_model.py      # CNN 模型定義
│   ├── model_utils.py    # 模型工具
│   └── fire_detection/   # 火災偵測模型檔案
├── utils/                 # 工具模組
│   ├── constants.py      # 常數定義
│   ├── file_utils.py     # 檔案工具
│   ├── text_utils.py     # 文字工具
│   └── security_utils.py # 安全工具
├── templates/
│   └── index.html        # Web 前端界面
├── static/
│   ├── css/              # 樣式檔案
│   ├── js/               # JavaScript 檔案
│   └── uploads/          # 上傳圖片目錄
├── knowledge_base/
│   └── sop.json         # SOP 知識庫
├── logs/                  # 日誌目錄
├── .env                   # 環境變數設定
└── .env.example          # 環境變數範例
```

## 快速開始

### 前置需求檢查

在開始之前，請確認您的系統滿足以下需求：

```bash
# 檢查 Python 版本 (需要 3.8+)
python --version

# 檢查可用記憶體 (建議 8GB+)
free -h  # Linux
# 或在 Windows: wmic memorychip get capacity

# 檢查可用儲存空間 (需要 5GB+)
df -h .  # Linux/Mac
# 或在 Windows: dir
```

### 1. 環境準備

#### 步驟 1.1：克隆專案
```bash
# 克隆專案到本地
git clone https://github.com/Lean0411/fire_report_project.git

# 進入專案目錄
cd fire_report_project
```

#### 步驟 1.2：建立虛擬環境
```bash
# 建立 Python 虛擬環境
python -m venv .venv

# 啟動虛擬環境
source .venv/bin/activate     # Linux/Mac
# 或
.venv\Scripts\activate        # Windows PowerShell
# 或  
.venv\Scripts\activate.bat    # Windows Command Prompt
```

#### 步驟 1.3：升級 pip 並安裝依賴
```bash
# 升級 pip 到最新版本
python -m pip install --upgrade pip

# 安裝專案依賴
pip install -r requirements.txt

# 驗證關鍵套件安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flask; print(f'Flask: {flask.__version__}')"
```

### 2. 模型設定

#### 步驟 2.1：檢查模型檔案
```bash
# 確認模型目錄存在
ls -la models/fire_detection/

# 檢查模型檔案 (應該存在 deep_wildfire_cnn_model_amp.pth)
ls -lh models/fire_detection/*.pth
```

#### 步驟 2.2：模型檔案下載 (如果缺失)
```bash
# 如果模型檔案不存在，請從以下位置下載：
# [模型下載連結] - 請聯絡專案維護者獲取模型檔案
# 檔案大小約：~421MB
```

### 3. 配置設定

#### 步驟 3.1：環境變數設定
```bash
# 複製環境變數範例檔案
cp .env.example .env

# 編輯環境變數 (使用您偏好的編輯器)
nano .env
# 或
vim .env
```

#### 步驟 3.2：設定內容說明
```bash
# Flask 應用安全金鑰 (重要：生產環境必須設置)
# 生成方式: python -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY=your_flask_secret_key_here

# OpenAI API 設定 (可選，需要 API Key)
# 從 https://platform.openai.com/api-keys 獲取
OPENAI_API_KEY=your_openai_api_key_here

# 本地語言引擎設定 (可選)
OLLAMA_HOST=http://127.0.0.1:11434  # Ollama 服務地址
OLLAMA_MODEL=gemma:7b              # 使用的模型

# 系統設定
FLASK_ENV=development              # 開發環境
FLASK_DEBUG=true                   # 啟用調試模式
PORT=5002                          # 服務端口

# 日誌設定
LOG_LEVEL=INFO                     # 日誌等級
```

### 4. 啟動系統

#### 步驟 4.1：啟動應用程式
```bash
# 使用自動啟動腳本 (推薦)
python run.py

# 或使用手動啟動
python app.py
```

#### 步驟 4.2：驗證啟動
```bash
# 檢查應用程式是否正常運行
curl http://127.0.0.1:5002

# 或在瀏覽器中訪問
# http://127.0.0.1:5002
# http://localhost:5002
```

### 5. 驗證安裝

#### 功能測試
1. **界面測試**：開啟瀏覽器訪問 http://127.0.0.1:5002
2. **上傳測試**：嘗試上傳一張測試圖片
3. **偵測測試**：查看是否能正常進行火災偵測

#### 常見啟動問題

**端口被佔用**
```bash
# 檢查端口使用情況
netstat -tulpn | grep :5002

# 更換端口啟動
PORT=5003 python run.py
```

**模型載入失敗**
```bash
# 檢查模型檔案權限
ls -la models/fire_detection/

# 檢查可用記憶體
free -h
```

**依賴套件錯誤**
```bash
# 重新安裝依賴
pip install --force-reinstall -r requirements.txt

# 清理 pip 快取
pip cache purge
```

### 🎉 安裝完成

系統成功啟動後，您將看到：
- 🌐 **Web 界面**：http://127.0.0.1:5002
- 📊 **控制台輸出**：顯示啟動資訊和日誌
- 📁 **日誌檔案**：`logs/app.log`

## 使用說明

### 基本使用流程
1. **選擇身份**：一般民眾/消防隊員/管理單位
2. **上傳圖片**：支援 JPG、PNG 格式，最大 5MB
3. **獲得分析**：系統自動進行火災檢測和智能分析
4. **查看建議**：根據身份獲得專業的應急處置建議

### API 端點

#### 火災檢測相關
- **`POST /api/detect`**：火災偵測主 API
- **`GET /api/detect/status`**：獲取檢測系統狀態

#### 安全建議相關
- **`GET /api/fire-safety-advice`**：獲取火災安全建議
- **`GET /api/safety/general-tips`**：獲取一般安全建議
- **`GET /api/safety/situation/<situation>`**：獲取特定情況建議
- **`GET /api/safety/emergency-contacts`**：獲取緊急聯絡方式
- **`GET /api/safety/checklist`**：獲取安全檢查清單
- **`POST /api/safety/role-advice`**：獲取角色化建議
- **`GET /api/safety/sop/validate`**：驗證 SOP 數據
- **`GET /api/safety/roles`**：獲取可用角色列表

#### Web 界面
- **`GET /`**：Web 界面首頁

## 使用範例

### 1. 基本 Web 界面使用

```bash
# 啟動系統後，在瀏覽器中訪問
http://127.0.0.1:5002

# 1. 選擇身份角色
# 2. 上傳火災圖片
# 3. 查看檢測結果和專業建議
```

### 2. API 程式化調用

#### 火災檢測 API

```python
import requests

# 準備上傳檔案和參數
files = {'file': open('fire_image.jpg', 'rb')}
data = {
    'role': 'firefighter',    # 身份：general/firefighter/management
    'use_ai': 'true',         # 啟用 AI 分析
    'ai_provider': 'openai'   # AI 提供者：openai/ollama
}

# 發送檢測請求
response = requests.post('http://127.0.0.1:5002/api/detect', 
                        files=files, data=data)

# 處理回應
if response.status_code == 200:
    result = response.json()
    if result['success']:
        detection = result['data']['detection']
        print(f"火災檢測: {'是' if detection['is_fire'] else '否'}")
        print(f"火災機率: {detection['fire_probability']}%")
        print(f"AI 分析: {result['data']['llm_report']}")
    else:
        print(f"錯誤: {result['error']}")
```

#### 獲取火災安全建議

```python
import requests

# 獲取特定角色的安全建議
response = requests.get('http://127.0.0.1:5002/api/fire-safety-advice',
                       params={'role': 'general'})

if response.status_code == 200:
    advice = response.json()
    for category, actions in advice.items():
        print(f"\n{category}:")
        for action in actions:
            print(f"- {action}")
```

#### JavaScript 前端整合

```javascript
// 檔案上傳和火災檢測
async function detectFire(imageFile, userRole) {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('role', userRole);
    formData.append('use_ai', 'true');
    formData.append('ai_provider', 'openai');
    
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            const detection = result.data.detection;
            console.log('檢測結果:', detection.is_fire ? '火災' : '安全');
            console.log('信心度:', detection.fire_probability + '%');
            return result.data;
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('檢測失敗:', error);
        throw error;
    }
}

// 使用範例
const fileInput = document.getElementById('file-input');
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        try {
            const result = await detectFire(file, 'general');
            displayResult(result);
        } catch (error) {
            alert('檢測失敗: ' + error.message);
        }
    }
});
```

## API 文檔

### 🔥 火災檢測 API

#### POST /api/detect - 火災檢測

**功能**: 上傳圖片進行火災檢測和智能分析

**請求格式**: `multipart/form-data`

**參數**:
| 參數名 | 類型 | 必填 | 說明 |
|--------|------|------|------|
| `file` | File | ✓ | 圖片檔案 (JPG/PNG/JPEG, 最大5MB) |
| `role` | String | ✓ | 使用者角色 (`general`/`firefighter`/`management`) |
| `use_ai` | String | - | 是否啟用AI分析 (`true`/`false`, 預設: `false`) |
| `ai_provider` | String | - | AI提供者 (`openai`/`ollama`, 預設: `openai`) |

**安全性**: 所有輸入參數都經過驗證和清理，防止 XSS 攻擊

**回應格式**: `application/json`

**成功回應** (200):
```json
{
  "success": true,
  "data": {
    "detection": {
      "is_fire": true,
      "fire_probability": 87.5,
      "no_fire_probability": 12.5,
      "model_confidence": 0.875
    },
    "filename": "annotated_20250617_123456.jpg",
    "recommendations": {
      "emergency_action_plan": [
        "制訂並訓練緊急應變計畫，包含疏散程序與責任分工",
        "定期進行疏散演練並檢視逃生路線標示"
      ]
    },
    "llm_report": "根據影像分析，檢測到明顯火焰和煙霧特徵...",
    "processing_time": 2.34
  }
}
```

**錯誤回應** (400/500):
```json
{
  "success": false,
  "error": "檔案格式不支援，請上傳 JPG/PNG/JPEG 格式"
}
```

#### GET /api/detect/status - 系統狀態

**功能**: 獲取檢測系統的運行狀態

**回應格式**: `application/json`

**成功回應** (200):
```json
{
  "success": true,
  "data": {
    "model": {
      "loaded": true,
      "device": "cpu",
      "model_path": "models/fire_detection/..."
    },
    "device": {
      "type": "cpu",
      "available_memory": "7.8GB"
    },
    "ai_services": {
      "openai_configured": true,
      "ollama_configured": false
    },
    "supported_roles": {
      "general": "一般民眾",
      "firefighter": "消防隊員",
      "management": "管理單位"
    },
    "supported_ai_providers": {
      "openai": "OpenAI GPT",
      "ollama": "Ollama 本地模型"
    }
  }
}
```

### 🛡️ 安全建議 API

#### GET /api/fire-safety-advice - 火災安全建議

**功能**: 獲取基於角色的火災安全建議

**請求格式**: `application/x-www-form-urlencoded` 或 `JSON`

**參數**:
| 參數名 | 類型 | 必填 | 說明 |
|--------|------|------|------|
| `role` | String | - | 使用者角色 (`general`/`firefighter`/`management`) |

**回應格式**: `application/json`

**成功回應** (200):
```json
{
  "emergency_action_plan": [
    "制訂並訓練緊急應變計畫，包含疏散程序與責任分工",
    "定期進行疏散演練並檢視逃生路線標示"
  ],
  "evacuation_preparedness": [
    "預先確認工作、學校與社區的疏散路線與集合地點",
    "為弱勢族群／行動不便者準備專用協助措施"
  ]
}
```

#### 其他安全建議 API

**GET /api/safety/general-tips** - 一般安全建議  
**GET /api/safety/situation/<situation>** - 情況專用建議  
**GET /api/safety/emergency-contacts** - 緊急聯絡  
**GET /api/safety/checklist** - 安全檢查清單  
**POST /api/safety/role-advice** - 角色化建議  
**GET /api/safety/sop/validate** - SOP 數據驗證  
**GET /api/safety/roles** - 角色列表  

### 🌐 Web 界面

#### GET / - Web 界面

**功能**: 提供視覺化的火災檢測 Web 界面

**回應**: HTML 頁面，包含：
- 圖片上傳區域（拖拽支援）
- 角色選擇下拉選單
- 即時檢測結果顯示
- 專業建議展示區域

## 設定說明

### 🔐 安全性配置

系統實施了多層安全防護機制：

#### 輸入驗證
- 所有用戶輸入經過嚴格驗證和清理
- HTML 轉義防止 XSS 攻擊
- 文件類型和大小限制
- 角色和參數白名單驗證

#### API Key 管理
- 環境變數方式存儲敏感信息
- API Key 格式驗證
- 錯誤處理時不洩露敏感信息

#### 錯誤處理
- 統一錯誤處理機制
- 日誌信息長度限制
- 敏感數據遮蔽

### 環境變數 (.env)
```bash
# Flask 應用安全金鑰 (重要：生產環境必須設置)
# 生成方式: python -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY=your_flask_secret_key_here

# OpenAI API 設定 (可選，需要 API Key)
# 從 https://platform.openai.com/api-keys 獲取
OPENAI_API_KEY=your_openai_api_key_here

# 本地語言引擎設定 (可選)
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gemma:7b

# 系統設定
FLASK_ENV=development
FLASK_DEBUG=true
PORT=5002

# 日誌設定
LOG_LEVEL=INFO
```

#### 🚨 安全警告
- **生產環境**必須設置 `FLASK_SECRET_KEY`
- **不要**在代碼中硬編碼 API Key
- **定期**更換 API Key
- **啟用 HTTPS** 在生產環境中

### 角色建議系統
系統根據使用者身份提供不同層級的建議：

- **一般民眾**：基礎逃生指導、緊急聯絡方式
- **消防隊員**：戰術評估、器材配置、安全協議
- **管理單位**：資源調度、應急管理、公眾溝通

## 技術特色

### 1. 深度學習檢測
- 使用自訓練的 CNN 模型
- 支援多種火災場景識別
- 高準確率的二元分類（火災/非火災）

### 2. 本地智能分析
- 整合 Ollama 本地大語言模型
- 無需外部 API 依賴
- 保護資料隱私安全

### 3. 專業知識庫
- 內建標準作業程序（SOP）
- 分角色專業建議系統
- 可擴展的知識庫架構

### 4. 現代化界面
- 響應式設計，支援多設備
- 拖拽上傳，操作便利
- 即時進度顯示
- 美觀的結果展示

## 系統需求

### 硬體需求
- **CPU**：多核心處理器推薦
- **RAM**：至少 8GB（模型載入需求）
- **儲存**：至少 5GB 可用空間

### 軟體需求
- **Python** 3.8+
- **PyTorch**
- **Flask**
- **Pillow**
- **其他依賴**：詳見 requirements.txt

## 故障排除

## 🚀 部署指南

### 生產環境部署

#### Docker 部署 (推薦)
```bash
# 建置 Docker 鏡像
docker build -t fire-detection .

# 運行容器
docker run -d \
  --name fire-detection-app \
  -p 5002:5002 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  fire-detection
```

#### 傳統部署
```bash
# 使用 systemd 服務
sudo tee /etc/systemd/system/fire-detection.service > /dev/null <<EOF
[Unit]
Description=Fire Detection System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/fire_report_project
Environment=PATH=/var/www/fire_report_project/.venv/bin
ExecStart=/var/www/fire_report_project/.venv/bin/python run.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 啟動服務
sudo systemctl enable fire-detection
sudo systemctl start fire-detection
```

#### Nginx 反向代理配置
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 10M;
    
    location / {
        proxy_pass http://127.0.0.1:5002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
```

### 效能優化

#### 模型優化
```bash
# 使用 GPU 加速 (如果可用)
# 在 config/settings.py 中設置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型量化 (減少內存使用)
# 在模型載入時啟用
model = torch.jit.script(model)  # TorchScript
```

#### 緩存配置
```python
# Redis 緩存 (可選)
# 在 requirements.txt 中添加: redis
# 在 config/settings.py 中配置
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
CACHE_TIMEOUT = 300  # 5 分鐘
```

## 🏢 技術架構

### 系統架構圖
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Web 前端   │   │  Flask API  │   │  業務服務  │
│  (React/JS)  │──▶│   路由層   │──▶│   處理層   │
└─────────────┘   └─────────────┘   └─────────────┘
                                         │
                           └──────────────┴───────────────┐
                                                         │
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   AI 服務   │   │  模型管理  │   │  SOP 知識  │
│ OpenAI/Ollama │◀──│   CNN模型   │◀──│   庫管理   │
└─────────────┘   └─────────────┘   └─────────────┘
```

### 核心模組說明

#### 1. API 層 (api/)
- **detection.py**: 火災檢測 API 端點
- **safety.py**: 安全建議 API 端點

#### 2. 服務層 (services/)
- **ai_service.py**: AI 模型整合 (OpenAI/Ollama)
- **image_service.py**: 圖片處理和標註
- **sop_service.py**: SOP 知識庫管理
- **safety_service.py**: 安全建議生成

#### 3. 模型層 (models/)
- **cnn_model.py**: CNN 模型定義
- **model_utils.py**: 模型加載和管理

#### 4. 工具層 (utils/)
- **security_utils.py**: 安全驗證和清理
- **file_utils.py**: 檔案處理工具
- **constants.py**: 常數定義

### 技術特色

#### 1. 模組化設計
- 清晰的分層架構
- 高度可維護性
- 易於擴展和測試

#### 2. 安全性設計
- 多層輸入驗證
- XSS 攻擊防護
- API Key 安全管理

#### 3. 效能優化
- 懶加載模型
- 異步處理機制
- 內存管理優化

#### 4. 擴展性
- 微服務架構就緒
- Docker 容器化支援
- 水平擴展能力

## 系統需求

### 最低需求
| 項目 | 規格 |
|------|------|
| **CPU** | 4 核心 2.0GHz+ |
| **RAM** | 8GB |
| **儲存** | 5GB 可用空間 |
| **Python** | 3.8+ |
| **作業系統** | Ubuntu 20.04+ / CentOS 8+ / Windows 10+ |

### 推薦配置
| 項目 | 規格 |
|------|------|
| **CPU** | 8 核心 3.0GHz+ |
| **RAM** | 16GB+ |
| **GPU** | NVIDIA GTX 1060+ (可選) |
| **儲存** | SSD 20GB+ |
| **網路** | 100Mbps+ |

### 軟體依賴
```python
# 核心依賴
Flask>=2.0.0
PyTorch>=2.0.0
Pillow>=9.0.0
requests>=2.28.0
python-dotenv>=0.19.0

# AI 服務
openai>=1.0.0  # 可選
ollama>=0.1.0  # 可選

# 其他工具
numpy>=1.21.0
opencv-python>=4.5.0
```

## 故障排除

### 常見問題

#### 1. 模型載入失敗
```bash
# 檢查模型檔案
ls -la models/fire_detection/
file models/fire_detection/*.pth

# 檢查檔案權限
chmod 644 models/fire_detection/*.pth

# 檢查內存使用
free -h
htop
```

#### 2. 端口被佔用
```bash
# 查看端口使用
netstat -tulpn | grep :5002
lsof -i :5002

# 結束佔用程序
sudo kill -9 <PID>

# 使用其他端口
PORT=5003 python run.py
```

#### 3. 圖片上傳失敗
```bash
# 檢查上傳目錄權限
ls -la static/uploads/
chmod 755 static/uploads/

# 檢查磁碟空間
df -h .

# 清理舊檔案
find static/uploads/ -mtime +7 -type f -delete
```

#### 4. AI 服務無回應
```bash
# 檢查 Ollama 服務
curl http://127.0.0.1:11434/api/version

# 重啟 Ollama
sudo systemctl restart ollama

# 檢查 OpenAI API Key
echo $OPENAI_API_KEY | cut -c1-10
```

### 日誌查看

#### 實時日誌監控
```bash
# 實時查看日誌
tail -f logs/app.log

# 篩選錯誤日誌
grep -i error logs/app.log | tail -20

# 查看特定時間範圍
grep "2025-06-17" logs/app.log
```

#### 日誌輪轉配置
```bash
# 使用 logrotate
sudo tee /etc/logrotate.d/fire-detection > /dev/null <<EOF
/var/www/fire_report_project/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
```

### 監控和告警

#### 系統監控
```bash
# 內存使用監控
ps aux | grep python | grep fire

# CPU 使用監控
top -p $(pgrep -f "python.*run.py")

# 磁碟使用監控
du -sh logs/ static/uploads/
```

#### 自動化監控腳本
```bash
#!/bin/bash
# health_check.sh
URL="http://127.0.0.1:5002/api/detect/status"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $URL)

if [ $RESPONSE -ne 200 ]; then
    echo "系統異常：HTTP $RESPONSE"
    # 發送告警郵件或通知
fi
```

## 🤝 貢獻指南

我們歡迎社區貢獻！以下是參與方式：

### 問題回報
- 使用 [GitHub Issues](https://github.com/Lean0411/fire_report_project/issues) 回報 Bug
- 提供詳細的錯誤說明和重現步驟
- 附上相關日誌和環境資訊

### 功能建議
- 在 Issues 中提出新功能建議
- 詳細描述功能需求和使用場景
- 提供模擬圖或原型（如有）

### 代碼貢獻
1. Fork 本倉庫
2. 建立特性分支：`git checkout -b feature/amazing-feature`
3. 提交修改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 開啟 Pull Request

### 開發指南
```bash
# 設置開發環境
git clone https://github.com/YOUR-USERNAME/fire_report_project.git
cd fire_report_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 開發工具

# 運行測試
pytest tests/

# 代碼格式化
black .
flake8 .
```

### 貢獻指引
- 遵循 PEP 8 代碼風格
- 為新功能添加測試
- 更新相關文檔
- 保持向後相容性

## 📜 授權條款

本專案採用 MIT 授權條款，詳情請查看 [LICENSE](LICENSE) 檔案。

### 簡單說明
- ✅ 商用使用
- ✅ 修改和分發
- ✅ 私人使用
- ✅ 包含在更大的作品中
- ❌ 無責任和無保證

---

## 🔗 相關連結

- **專案倉庫**: [GitHub](https://github.com/Lean0411/fire_report_project)
- **問題回報**: [Issues](https://github.com/Lean0411/fire_report_project/issues)
- **功能建議**: [Discussions](https://github.com/Lean0411/fire_report_project/discussions)
- **文檔網站**: [Wiki](https://github.com/Lean0411/fire_report_project/wiki)

## 📞 支援與聯絡

如果您在使用過程中遇到問題，請：

1. 查閱 [故障排除](#故障排除) 節
2. 搜索現有 [Issues](https://github.com/Lean0411/fire_report_project/issues)
3. 建立新的 Issue 並提供詳細資訊

**感謝您的使用和貢獻！**

---

🔥 **重要提醒**：本系統僅供輔助參考，實際火災情況請優先確保人身安全並立即聯絡消防單位。
