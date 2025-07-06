# API 響應格式統一化遷移指南

## 概述

本指南說明如何將現有的 API 端點遷移到新的統一響應格式。

## 統一響應格式結構

### 成功響應
```json
{
  "status": "success",
  "message": "操作成功",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    // 實際數據
  },
  "meta": {
    // 可選的元數據，如分頁信息
  }
}
```

### 錯誤響應
```json
{
  "status": "error",
  "message": "錯誤訊息",
  "timestamp": "2024-01-01T12:00:00Z",
  "error": {
    "code": "ERROR_CODE",
    "details": {
      // 詳細錯誤信息
    }
  },
  "trace_id": "用於追蹤的 ID（可選）"
}
```

## 快速開始

### 1. 導入必要的工具

```python
from utils.api_response import APIResponse
from utils.decorators import handle_api_errors, validate_request_json
```

### 2. 使用裝飾器處理錯誤

```python
@app.route('/api/endpoint', methods=['POST'])
@handle_api_errors  # 自動處理異常
@validate_request_json(YourSchema)  # 自動驗證請求
def your_endpoint():
    data = request.validated_data
    # 你的業務邏輯
    return APIResponse.success(data=result)
```

## 遷移示例

### 原始代碼
```python
@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data.get('username'):
            return jsonify({'error': 'Username required'}), 400
        
        # 登入邏輯
        if login_success:
            return jsonify({
                'user': user_data,
                'token': token
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 遷移後的代碼
```python
@auth_bp.route('/login', methods=['POST'])
@handle_api_errors
@validate_request_json(LoginSchema)
def login():
    data = request.validated_data
    
    # 登入邏輯
    if login_success:
        return APIResponse.success(
            data={
                'user': user_data,
                'token': token
            },
            message="登入成功"
        )
    else:
        return APIResponse.unauthorized(
            message="用戶名或密碼錯誤"
        )
```

## 常用響應方法

### 成功響應
- `APIResponse.success()` - 通用成功響應 (200)
- `APIResponse.created()` - 創建成功 (201)
- `APIResponse.no_content()` - 無內容 (204)

### 錯誤響應
- `APIResponse.error()` - 通用錯誤響應
- `APIResponse.validation_error()` - 驗證錯誤 (422)
- `APIResponse.unauthorized()` - 未授權 (401)
- `APIResponse.forbidden()` - 禁止訪問 (403)
- `APIResponse.not_found()` - 資源不存在 (404)
- `APIResponse.conflict()` - 資源衝突 (409)
- `APIResponse.server_error()` - 服務器錯誤 (500)

### 特殊響應
- `APIResponse.paginated()` - 分頁數據響應

## 最佳實踐

### 1. 使用有意義的錯誤代碼
```python
return APIResponse.error(
    message="用戶名已存在",
    error_code="USERNAME_EXISTS",
    status_code=409
)
```

### 2. 提供詳細的驗證錯誤
```python
return APIResponse.validation_error(
    errors={
        "email": ["格式無效", "已被使用"],
        "password": ["長度不足"]
    }
)
```

### 3. 在分頁響應中包含元數據
```python
return APIResponse.paginated(
    data=items,
    page=1,
    per_page=20,
    total=100,
    extra_meta={
        "filters": request.args.to_dict()
    }
)
```

### 4. 使用裝飾器簡化代碼
```python
@handle_api_errors  # 處理所有異常
@validate_request_json(CreateUserSchema)  # 驗證請求數據
@measure_performance  # 測量性能
def create_user():
    # 簡潔的業務邏輯
    pass
```

## 遷移檢查清單

- [ ] 導入 APIResponse 和相關裝飾器
- [ ] 替換所有 `jsonify()` 調用為適當的 APIResponse 方法
- [ ] 添加 `@handle_api_errors` 裝飾器
- [ ] 使用 Schema 類進行請求驗證
- [ ] 統一錯誤代碼和訊息
- [ ] 更新 API 文檔
- [ ] 更新前端代碼以處理新的響應格式
- [ ] 測試所有端點

## 兼容性考慮

如果需要保持向後兼容，可以：

1. 創建新的版本化端點（如 `/api/v2/`）
2. 使用功能開關逐步遷移
3. 提供響應格式轉換中間件

## 示例項目

查看 `api/v1/auth_v2.py` 作為完整的遷移示例。