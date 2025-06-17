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
├── templates/
│   └── index.html        # Web 前端界面
├── static/
│   ├── css/              # 樣式檔案
│   ├── js/               # JavaScript 檔案
│   └── uploads/          # 上傳圖片目錄
├── models/
│   └── fire_detection/   # 火災偵測模型
├── knowledge_base/
│   └── sop.json         # SOP 知識庫
└── .env                  # 環境變數設定
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
# 本地語言引擎設定 (可選)
OLLAMA_HOST=http://127.0.0.1:11434  # Ollama 服務地址
OLLAMA_MODEL=gemma:7b              # 使用的模型

# OpenAI 設定 (可選，需要 API Key)
OPENAI_API_KEY=your_api_key_here   # 替換為您的 OpenAI API Key

# 系統設定
FLASK_ENV=development              # 開發環境
FLASK_DEBUG=true                   # 啟用調試模式
PORT=5002                          # 服務端口
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
- **`POST /api/detect`**：火災偵測主 API
- **`GET /api/fire-safety-advice`**：獲取火災安全建議
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

### POST /api/detect - 火災檢測

**功能**: 上傳圖片進行火災檢測和智能分析

**請求格式**: `multipart/form-data`

**參數**:
| 參數名 | 類型 | 必填 | 說明 |
|--------|------|------|------|
| `file` | File | ✓ | 圖片檔案 (JPG/PNG/JPEG, 最大5MB) |
| `role` | String | ✓ | 使用者角色 (`general`/`firefighter`/`management`) |
| `use_ai` | String | - | 是否啟用AI分析 (`true`/`false`, 預設: `false`) |
| `ai_provider` | String | - | AI提供者 (`openai`/`ollama`, 預設: `openai`) |

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

### GET /api/fire-safety-advice - 火災安全建議

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

### GET / - Web 界面

**功能**: 提供視覺化的火災檢測 Web 界面

**回應**: HTML 頁面，包含：
- 圖片上傳區域（拖拽支援）
- 角色選擇下拉選單
- 即時檢測結果顯示
- 專業建議展示區域

## 設定說明

### 環境變數 (.env)
```bash
# 本地語言引擎設定
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gemma:7b

# 系統設定
FLASK_ENV=development
FLASK_DEBUG=true
```

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

### 常見問題
1. **模型載入失敗**：確認模型檔案路徑正確
2. **端口被佔用**：檢查 5002 端口是否可用
3. **圖片上傳失敗**：確認檔案格式和大小限制
4. **語言引擎無回應**：確認 Ollama 服務運行狀態

### 日誌查看
系統日誌保存在 `logs/app.log`，包含詳細的運行資訊和錯誤記錄。

## 貢獻指南

歡迎提交 Issue 和 Pull Request 來改善系統功能。

## 授權條款

本專案採用開源授權，詳情請查看 LICENSE 檔案。

---

🔥 **重要提醒**：本系統僅供輔助參考，實際火災情況請優先確保人身安全並立即聯絡消防單位。
