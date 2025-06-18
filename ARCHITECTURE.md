# FireGuard AI - 新架構說明

## 🏗️ 架構概覽

系統已從原本的單體架構重構為模組化的企業級架構，具備完整的認證、快取、監控等功能。

## 📁 新增目錄結構

```
fire_report_project/
├── data/                    # 資料層
│   ├── models/             # 資料模型
│   │   ├── user_model.py
│   │   ├── detection_history.py
│   │   └── system_metrics.py
│   ├── repositories/       # 資料存取層
│   │   ├── user_repository.py
│   │   └── detection_repository.py
│   └── database.py         # 資料庫配置
├── services/               # 服務層
│   ├── auth/              # 認證服務
│   │   └── auth_service.py
│   └── cache_service.py   # 快取服務
├── api/                   # API層
│   ├── v1/                # API v1版本
│   │   ├── auth.py        # 認證端點
│   │   ├── detection.py   # 檢測端點
│   │   └── monitoring.py  # 監控端點
│   └── middleware/        # 中介軟體
│       ├── rate_limit.py  # 速率限制
│       └── metrics.py     # 指標收集
└── monitoring/            # 監控系統
    └── health_checks.py   # 健康檢查
```

## 🚀 新功能特色

### 1. 認證與授權
- JWT Token 認證
- API Key 支援
- 角色權限控制 (admin, firefighter, user)
- 密碼強度驗證

### 2. 資料庫持久化
- SQLAlchemy ORM
- 用戶管理
- 檢測歷史記錄
- 系統指標存儲

### 3. 快取系統
- Redis 支援（可選）
- 記憶體快取備援
- 模型預測快取
- SOP 回應快取

### 4. 監控與健康檢查
- 系統資源監控
- API 性能指標
- 健康狀態檢查
- 錯誤率追蹤

### 5. API 版本管理
- 新版 API (v1) 支援
- 舊版 API 向後兼容
- 統一錯誤處理

## 🔧 快速啟動

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 環境變數配置
創建 `.env` 檔案：
```env
# 資料庫
DATABASE_URL=sqlite:///fire_guard.db

# JWT 密鑰
JWT_SECRET_KEY=your-secret-key-here

# Redis (可選)
REDIS_URL=redis://localhost:6379/0

# 管理員帳號
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@fireguard.ai
ADMIN_PASSWORD=Admin123!

# AI 服務
OPENAI_API_KEY=your-openai-key
```

### 3. 啟動應用
```bash
python app.py
```

## 📚 API 使用指南

### 認證
```bash
# 註冊用戶
curl -X POST http://localhost:5002/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"user1","email":"user@example.com","password":"Password123!"}'

# 登入
curl -X POST http://localhost:5002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username_or_email":"user1","password":"Password123!"}'
```

### 火災檢測
```bash
# 檢測圖片
curl -X POST http://localhost:5002/api/v1/detection/analyze \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "image=@fire_image.jpg"
```

### 監控
```bash
# 健康檢查
curl http://localhost:5002/api/v1/monitoring/health

# 系統指標 (需管理員權限)
curl -H "Authorization: Bearer ADMIN_JWT_TOKEN" \
  http://localhost:5002/api/v1/monitoring/metrics/system
```

## 🛡️ 安全功能

1. **密碼安全**: 強制密碼複雜度要求
2. **JWT Token**: 安全的無狀態認證
3. **速率限制**: 防止 API 濫用
4. **安全標頭**: XSS、CSRF 防護
5. **輸入驗證**: 全面的請求驗證

## 📊 監控功能

1. **健康檢查**: 資料庫、快取、AI 服務狀態
2. **性能指標**: API 回應時間、錯誤率
3. **系統監控**: CPU、記憶體、磁碟使用
4. **檢測統計**: 火災檢測趨勢分析

## 🔄 向後兼容

舊版 API 端點仍然可用：
- `/api/detect` -> 原檢測端點
- `/api/safety` -> 原安全建議端點

新版 API 提供更豐富的功能：
- `/api/v1/detection/analyze` -> 增強檢測端點
- `/api/v1/monitoring/*` -> 監控端點

## 📈 效能改進

1. **快取機制**: 減少重複計算
2. **資料庫索引**: 優化查詢性能
3. **連接池**: 資料庫連接管理
4. **非同步處理**: 背景監控任務

## 🚀 部署建議

### 開發環境
```bash
python app.py
```

### 生產環境
```bash
gunicorn --bind 0.0.0.0:5002 --workers 4 app:app
```

### Docker 部署
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5002
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]
```

## 🔮 未來擴展

1. **微服務架構**: 可拆分為獨立服務
2. **消息隊列**: Celery/RQ 背景任務
3. **API 網關**: 統一入口點
4. **容器編排**: Kubernetes 部署
5. **CI/CD 流水線**: 自動化部署

這個重構後的架構為系統提供了企業級的可擴展性、可維護性和可靠性基礎。