# FireGuard AI 測試框架

## 概述

本測試框架為 FireGuard AI 專案提供完整的測試覆蓋，包括單元測試和整合測試。

## 目錄結構

```
tests/
├── conftest.py                    # pytest配置和共享fixtures
├── pytest.ini                    # pytest設定檔
├── README.md                      # 本文件
├── unit/                          # 單元測試
│   ├── models/
│   │   ├── test_simple.py        # 基礎測試（無外部依賴）
│   │   └── test_cnn_model.py     # CNN模型測試（需要torch）
│   ├── services/
│   │   └── test_ai_service.py    # AI服務測試（需要openai）
│   ├── utils/
│   │   └── test_security_utils.py # 安全工具測試
│   └── api/
│       └── test_detection.py     # API檢測端點測試（需要flask）
├── integration/                   # 整合測試
│   ├── test_api_endpoints.py     # API端點整合測試
│   └── test_complete_workflow.py # 完整工作流程測試
└── fixtures/                     # 測試數據
```

## 執行測試

### 1. 基礎測試（推薦）

運行不需要外部依賴的測試：

```bash
# 運行基礎測試
python3 -m pytest tests/unit/models/test_simple.py tests/unit/utils/test_security_utils.py -v

# 運行特定測試
python3 -m pytest tests/unit/utils/test_security_utils.py::TestStringInput -v
```

### 2. 完整測試（需要安裝依賴）

如果已安裝所有依賴（flask, torch, openai等）：

```bash
# 運行所有單元測試
python3 -m pytest tests/unit/ -v

# 運行所有測試
python3 -m pytest -v

# 運行測試並生成覆蓋率報告
python3 -m pytest --cov=. --cov-report=html
```

### 3. 安裝測試依賴

```bash
# 安裝基本依賴
pip install pytest pytest-cov pytest-mock

# 安裝完整依賴
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

## 故障排除

### 常見問題

1. **ModuleNotFoundError**: 缺少依賴模組
   ```bash
   pip install -r requirements.txt
   ```

2. **Import錯誤**: Python路徑問題
   ```bash
   export PYTHONPATH=/home/lean/fire_report_project:$PYTHONPATH
   ```

3. **權限錯誤**: 檔案存取權限
   ```bash
   chmod +x run_tests.sh
   ```

### 跳過特定測試

```bash
# 跳過需要網路的測試
python3 -m pytest -m "not network"

# 跳過需要GPU的測試
python3 -m pytest -m "not gpu"
```

## 持續改進

- [ ] 添加更多邊界案例測試
- [ ] 提高測試覆蓋率到90%+
- [ ] 添加性能基準測試
- [ ] 整合CI/CD自動化測試
- [ ] 添加回歸測試套件