"""
測試 API 響應格式統一化工具
"""
import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime
from utils.api_response import (
    APIResponse, ResponseStatus, handle_exception, 
    format_validation_errors
)


class TestAPIResponse:
    """APIResponse 測試類"""
    
    def test_success_response(self):
        """測試成功響應"""
        data = {"id": 1, "name": "Test"}
        response, status_code = APIResponse.success(data=data)
        
        # 解析響應
        response_data = json.loads(response.data)
        
        assert status_code == 200
        assert response_data["status"] == "success"
        assert response_data["message"] == "操作成功"
        assert response_data["data"] == data
        assert "timestamp" in response_data
    
    def test_success_response_custom_message(self):
        """測試自定義訊息的成功響應"""
        response, status_code = APIResponse.success(
            data={"result": "ok"},
            message="自定義成功訊息",
            status_code=201
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 201
        assert response_data["message"] == "自定義成功訊息"
    
    def test_success_response_with_meta(self):
        """測試帶元數據的成功響應"""
        meta = {"version": "1.0", "request_id": "123"}
        response, status_code = APIResponse.success(
            data={"test": "data"},
            meta=meta
        )
        
        response_data = json.loads(response.data)
        
        assert response_data["meta"] == meta
    
    def test_error_response(self):
        """測試錯誤響應"""
        response, status_code = APIResponse.error(
            message="發生錯誤",
            errors={"field": ["錯誤1", "錯誤2"]},
            error_code="TEST_ERROR",
            status_code=400
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 400
        assert response_data["status"] == "error"
        assert response_data["message"] == "發生錯誤"
        assert response_data["error"]["code"] == "TEST_ERROR"
        assert response_data["error"]["details"] == {"field": ["錯誤1", "錯誤2"]}
    
    def test_error_response_default_code(self):
        """測試默認錯誤代碼"""
        response, status_code = APIResponse.error(status_code=500)
        
        response_data = json.loads(response.data)
        
        assert response_data["error"]["code"] == "ERR_500"
    
    def test_error_response_with_trace_id(self):
        """測試帶追蹤 ID 的錯誤響應"""
        response, status_code = APIResponse.error(
            message="錯誤",
            trace_id="trace-123"
        )
        
        response_data = json.loads(response.data)
        
        assert response_data["trace_id"] == "trace-123"
    
    def test_paginated_response(self):
        """測試分頁響應"""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        response, status_code = APIResponse.paginated(
            data=data,
            page=2,
            per_page=10,
            total=25
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 200
        assert response_data["data"] == data
        assert response_data["meta"]["pagination"]["page"] == 2
        assert response_data["meta"]["pagination"]["per_page"] == 10
        assert response_data["meta"]["pagination"]["total"] == 25
        assert response_data["meta"]["pagination"]["total_pages"] == 3
        assert response_data["meta"]["pagination"]["has_prev"] is True
        assert response_data["meta"]["pagination"]["has_next"] is True
    
    def test_paginated_response_first_page(self):
        """測試第一頁的分頁響應"""
        response, status_code = APIResponse.paginated(
            data=[],
            page=1,
            per_page=10,
            total=5
        )
        
        response_data = json.loads(response.data)
        pagination = response_data["meta"]["pagination"]
        
        assert pagination["has_prev"] is False
        assert pagination["has_next"] is False
    
    def test_paginated_response_with_extra_meta(self):
        """測試帶額外元數據的分頁響應"""
        extra_meta = {"filters": {"status": "active"}}
        response, status_code = APIResponse.paginated(
            data=[],
            page=1,
            per_page=10,
            total=0,
            extra_meta=extra_meta
        )
        
        response_data = json.loads(response.data)
        
        assert "filters" in response_data["meta"]
        assert response_data["meta"]["filters"] == {"status": "active"}
    
    def test_created_response(self):
        """測試創建成功響應"""
        data = {"id": 123, "name": "New Resource"}
        response, status_code = APIResponse.created(
            data=data,
            location="/api/resources/123"
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 201
        assert response_data["message"] == "創建成功"
        assert response_data["data"] == data
        assert response.headers.get("Location") == "/api/resources/123"
    
    def test_no_content_response(self):
        """測試無內容響應"""
        response, status_code = APIResponse.no_content()
        
        assert status_code == 204
    
    def test_validation_error_response(self):
        """測試驗證錯誤響應"""
        errors = {
            "email": ["格式無效", "已存在"],
            "password": ["太短"]
        }
        response, status_code = APIResponse.validation_error(errors=errors)
        
        response_data = json.loads(response.data)
        
        assert status_code == 422
        assert response_data["error"]["code"] == "VALIDATION_ERROR"
        assert response_data["error"]["details"] == errors
    
    def test_unauthorized_response(self):
        """測試未授權響應"""
        response, status_code = APIResponse.unauthorized()
        
        response_data = json.loads(response.data)
        
        assert status_code == 401
        assert response_data["message"] == "未授權"
        assert response_data["error"]["code"] == "UNAUTHORIZED"
    
    def test_forbidden_response(self):
        """測試禁止訪問響應"""
        response, status_code = APIResponse.forbidden()
        
        response_data = json.loads(response.data)
        
        assert status_code == 403
        assert response_data["message"] == "無權限"
        assert response_data["error"]["code"] == "FORBIDDEN"
    
    def test_not_found_response(self):
        """測試資源不存在響應"""
        response, status_code = APIResponse.not_found(resource="用戶")
        
        response_data = json.loads(response.data)
        
        assert status_code == 404
        assert response_data["message"] == "用戶不存在"
        assert response_data["error"]["code"] == "NOT_FOUND"
    
    def test_not_found_response_custom_message(self):
        """測試自定義訊息的資源不存在響應"""
        response, status_code = APIResponse.not_found(
            message="找不到指定的檔案"
        )
        
        response_data = json.loads(response.data)
        
        assert response_data["message"] == "找不到指定的檔案"
    
    def test_method_not_allowed_response(self):
        """測試方法不允許響應"""
        response, status_code = APIResponse.method_not_allowed(
            allowed_methods=["GET", "POST"]
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 405
        assert response_data["error"]["code"] == "METHOD_NOT_ALLOWED"
        assert response_data["error"]["details"]["allowed_methods"] == ["GET", "POST"]
    
    def test_conflict_response(self):
        """測試資源衝突響應"""
        response, status_code = APIResponse.conflict(
            message="郵箱已被使用"
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 409
        assert response_data["message"] == "郵箱已被使用"
        assert response_data["error"]["code"] == "CONFLICT"
    
    @patch('utils.api_response.current_app')
    def test_server_error_response(self, mock_app):
        """測試服務器錯誤響應"""
        mock_app.debug = False
        
        response, status_code = APIResponse.server_error(
            error_id="error-123"
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 500
        assert response_data["error"]["code"] == "INTERNAL_SERVER_ERROR"
        assert response_data["error"]["details"]["error_id"] == "error-123"
        assert "debug" not in response_data["error"]["details"]
    
    @patch('utils.api_response.current_app')
    def test_server_error_response_with_debug(self, mock_app):
        """測試帶調試信息的服務器錯誤響應"""
        mock_app.debug = True
        
        debug_info = {"exception": "Test error", "type": "TestException"}
        response, status_code = APIResponse.server_error(
            debug_info=debug_info
        )
        
        response_data = json.loads(response.data)
        
        assert "debug" in response_data["error"]["details"]
        assert response_data["error"]["details"]["debug"] == debug_info
    
    def test_service_unavailable_response(self):
        """測試服務不可用響應"""
        response, status_code = APIResponse.service_unavailable(
            retry_after=60
        )
        
        response_data = json.loads(response.data)
        
        assert status_code == 503
        assert response_data["error"]["code"] == "SERVICE_UNAVAILABLE"
        assert response.headers.get("Retry-After") == "60"


class TestExceptionHandler:
    """異常處理器測試類"""
    
    @patch('utils.api_response.current_app')
    @patch('utils.api_response.uuid.uuid4')
    def test_handle_exception(self, mock_uuid, mock_app):
        """測試異常處理"""
        mock_uuid.return_value = "test-error-id"
        mock_app.debug = False
        mock_app.logger.error = Mock()
        
        exception = ValueError("測試錯誤")
        response, status_code = handle_exception(exception)
        
        response_data = json.loads(response.data)
        
        assert status_code == 500
        assert response_data["message"] == "服務器發生錯誤，請稍後重試"
        assert response_data["error"]["details"]["error_id"] == "test-error-id"
        
        # 確認錯誤被記錄
        mock_app.logger.error.assert_called_once()
    
    @patch('utils.api_response.current_app')
    def test_handle_exception_with_debug(self, mock_app):
        """測試帶調試信息的異常處理"""
        mock_app.debug = True
        mock_app.logger.error = Mock()
        
        exception = ValueError("測試錯誤")
        response, status_code = handle_exception(exception)
        
        response_data = json.loads(response.data)
        
        assert "debug" in response_data["error"]["details"]
        assert response_data["error"]["details"]["debug"]["exception"] == "測試錯誤"
        assert response_data["error"]["details"]["debug"]["type"] == "ValueError"


class TestFormatValidationErrors:
    """驗證錯誤格式化測試類"""
    
    def test_format_simple_errors(self):
        """測試簡單錯誤格式化"""
        errors = {
            "email": "無效格式",
            "password": "太短"
        }
        
        formatted = format_validation_errors(errors)
        
        assert formatted == {
            "email": ["無效格式"],
            "password": ["太短"]
        }
    
    def test_format_list_errors(self):
        """測試列表錯誤格式化"""
        errors = {
            "email": ["格式無效", "已存在"],
            "age": ["必須大於0"]
        }
        
        formatted = format_validation_errors(errors)
        
        assert formatted == errors
    
    def test_format_nested_errors(self):
        """測試嵌套錯誤格式化"""
        errors = {
            "user": {
                "email": "無效",
                "profile": {
                    "name": "必填"
                }
            },
            "tags": ["至少選擇一個"]
        }
        
        formatted = format_validation_errors(errors)
        
        assert "user.email" in formatted
        assert "user.profile" in formatted
        assert formatted["user.email"] == ["無效"]
        assert "tags" in formatted
        assert formatted["tags"] == ["至少選擇一個"]
    
    def test_format_mixed_errors(self):
        """測試混合類型錯誤格式化"""
        errors = {
            "field1": "錯誤1",
            "field2": ["錯誤2", "錯誤3"],
            "field3": {
                "subfield": "錯誤4"
            }
        }
        
        formatted = format_validation_errors(errors)
        
        assert formatted["field1"] == ["錯誤1"]
        assert formatted["field2"] == ["錯誤2", "錯誤3"]
        assert formatted["field3.subfield"] == ["錯誤4"]