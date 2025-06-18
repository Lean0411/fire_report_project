# FireGuard AI 測試框架

## 🎯 概述

本測試框架為 FireGuard AI 專案提供完整的測試覆蓋，包括單元測試和整合測試。採用模組化設計，支援漸進式測試執行。

## 📁 目錄結構

```
tests/
├── 📋 conftest.py                 # pytest配置和共享fixtures
├── ⚙️  pytest.ini                 # pytest設定檔  
├── ⚙️  .coveragerc                # 覆蓋率配置
├── 📖 README.md                   # 本文件
├── 📖 TESTING.md                  # 測試指南
├── 🧪 unit/                       # 單元測試
│   ├── models/
│   │   ├── test_simple.py        # ✅ 基礎測試（無外部依賴）
│   │   └── test_cnn_model.py     # 🔥 CNN模型測試（需要torch）
│   ├── services/
│   │   └── test_ai_service.py    # 🤖 AI服務測試（需要openai）
│   ├── utils/
│   │   └── test_security_utils.py # ✅ 安全工具測試
│   └── api/
│       └── test_detection.py     # 🌐 API檢測端點測試（需要flask）
├── 🔗 integration/                # 整合測試
│   ├── test_api_endpoints.py     # 🌐 API端點整合測試
│   └── test_complete_workflow.py # 🔄 完整工作流程測試
└── 📦 fixtures/                   # 測試數據和輔助檔案
```

### 標記說明
- ✅ **無依賴測試** - 可獨立執行，速度快
- 🔥 **模型測試** - 需要 PyTorch/深度學習框架
- 🤖 **AI服務測試** - 需要 OpenAI/Ollama API
- 🌐 **Web測試** - 需要 Flask/Web框架
- 🔄 **整合測試** - 需要完整環境

## 🚀 快速開始

### 方法一：使用測試腳本（推薦）
```bash
# 從專案根目錄執行
./scripts/run_tests.sh
```

### 方法二：手動執行測試

#### 1. ✅ 基礎測試（無外部依賴）
```bash
# 運行核心測試套件
python3 -m pytest tests/unit/models/test_simple.py tests/unit/utils/test_security_utils.py -v

# 運行特定測試類別
python3 -m pytest tests/unit/utils/test_security_utils.py::TestStringInput -v

# 使用標記篩選
python3 -m pytest -m unit -v
```

#### 2. 🔥 完整測試（需要依賴）
```bash
# 運行所有單元測試
python3 -m pytest tests/unit/ -v

# 運行整合測試
python3 -m pytest tests/integration/ -v

# 運行所有測試
python3 -m pytest -v
```

#### 3. 📊 覆蓋率測試
```bash
# 生成覆蓋率報告
python3 -m pytest --cov=utils --cov=config --cov-report=html --cov-config=tests/.coveragerc

# 查看覆蓋率摘要
python3 -m pytest --cov=. --cov-report=term-missing
```

### 📦 依賴安裝

#### 基本測試依賴
```bash
pip install pytest pytest-cov pytest-mock pytest-flask
```

#### 完整開發環境
```bash
pip install -r requirements.txt
```

## 測試類別

### 單元測試

- **test_simple.py**: 基礎測試，驗證專案結構、配置和基本功能
- **test_security_utils.py**: 安全工具函數測試，包括輸入驗證和清理
- **test_cnn_model.py**: CNN模型測試（需要torch）
- **test_ai_service.py**: AI服務測試（需要openai）
- **test_detection.py**: API檢測端點測試（需要flask）

### 整合測試

- **test_api_endpoints.py**: 完整API端點測試
- **test_complete_workflow.py**: 端到端工作流程測試

## 測試覆蓋範圍

### ✅ 已測試模組

- 專案結構驗證
- 配置文件驗證
- 安全工具函數
- 字串輸入清理
- 角色和提供者驗證
- 機率和布爾值驗證
- 檔名和URL驗證
- API Key驗證

### 🚧 需要依賴的測試

- CNN模型載入和預測
- AI服務（OpenAI/Ollama）
- Flask API端點
- 圖像處理和上傳
- 完整工作流程

## 測試配置

### pytest.ini 配置

- 測試路徑: `tests/`
- 覆蓋率報告: HTML + 終端
- 警告過濾: 忽略第三方庫警告
- 標記系統: unit, integration, slow, ai, model

### 環境變數

測試時自動設置：
- `TESTING=True`
- `AI_PROVIDER=mock`

## 最佳實踐

1. **優先運行基礎測試**: 無需外部依賴，執行速度快
2. **使用標記**: 區分不同類型的測試
3. **Mock外部服務**: 避免測試依賴外部API
4. **清理測試資源**: 自動清理臨時文件
5. **詳細斷言**: 提供清晰的測試失敗信息

## 🔧 高級使用

### 測試標記系統
```bash
# 只運行單元測試
python3 -m pytest -m unit

# 只運行整合測試  
python3 -m pytest -m integration

# 跳過慢速測試
python3 -m pytest -m "not slow"

# 只運行AI相關測試
python3 -m pytest -m ai
```

### 並行測試執行
```bash
# 安裝pytest-xdist
pip install pytest-xdist

# 並行執行測試
python3 -m pytest -n auto
```

### 測試結果格式化
```bash
# JUnit XML格式（CI/CD）
python3 -m pytest --junitxml=test-results.xml

# JSON格式
pip install pytest-json-report
python3 -m pytest --json-report --json-report-file=test-results.json
```

## ⚠️ 故障排除

### 常見問題

1. **ModuleNotFoundError**: 缺少依賴模組
   ```bash
   pip install -r requirements.txt
   ```

2. **Import錯誤**: Python路徑問題
   ```bash
   export PYTHONPATH=$(pwd):$PYTHONPATH
   python3 -m pytest
   ```

3. **權限錯誤**: 測試腳本執行權限
   ```bash
   chmod +x scripts/run_tests.sh
   ```

4. **Flask App 未找到**: 確保在正確目錄
   ```bash
   cd /path/to/fire_report_project
   python3 -m pytest
   ```

### 調試技巧
```bash
# 詳細輸出（包含print語句）
python3 -m pytest -s -v

# 只運行失敗的測試
python3 -m pytest --lf

# 在第一個失敗時停止
python3 -m pytest -x

# 顯示最慢的10個測試
python3 -m pytest --durations=10
```

## 📈 覆蓋率目標

| 模組 | 當前覆蓋率 | 目標覆蓋率 | 狀態 |
|------|------------|------------|------|
| utils/security_utils.py | 95% | 98% | ✅ 優秀 |
| utils/constants.py | 100% | 100% | ✅ 完美 |
| config/settings.py | 58% | 80% | 🔄 改進中 |
| 整體專案 | 36% | 70% | 🎯 目標 |

## 🚀 持續改進計畫

### ✅ 已完成
- [x] 基礎測試框架建立
- [x] 單元測試覆蓋核心功能
- [x] 自動化測試腳本
- [x] 覆蓋率報告生成
- [x] 測試標記分類系統

### 🔄 進行中
- [ ] 提高測試覆蓋率到70%+
- [ ] 添加更多邊界案例測試
- [ ] 完善整合測試套件

### 🎯 計畫中
- [ ] 添加性能基準測試
- [ ] 整合CI/CD自動化測試  
- [ ] 添加回歸測試套件
- [ ] 視覺回歸測試（UI）
- [ ] 負載測試和壓力測試

## 📞 支援與貢獻

### 報告問題
如果發現測試相關問題，請在 GitHub Issues 中回報，並包含：
- 測試執行環境
- 錯誤訊息完整內容
- 重現步驟

### 貢獻測試
歡迎提交新的測試案例：
1. 遵循現有的測試結構
2. 添加適當的測試標記
3. 確保測試獨立且可重複執行
4. 更新相關文檔

---

📝 **最後更新**: 2024-06-18  
🔗 **相關文檔**: [TESTING.md](TESTING.md) | [主要README](../README.md)