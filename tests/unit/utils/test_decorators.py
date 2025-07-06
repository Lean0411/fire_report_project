"""
測試裝飾器
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from flask import Flask, request
from marshmallow import Schema, fields, ValidationError
from utils.decorators import (
    handle_api_errors, validate_request_json, validate_query_params,
    measure_performance, require_json_content_type, log_request_response
)
from utils.api_response import APIResponse
import json


# 測試用的 Schema
class TestSchema(Schema):
    name = fields.Str(required=True)
    age = fields.Int(required=True, validate=lambda x: x > 0)


class TestDecorators:
    """裝飾器測試類"""
    
    @pytest.fixture
    def app(self):
        """創建測試 Flask 應用"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """創建測試客戶端"""
        return app.test_client()
    
    def test_handle_api_errors_success(self, app):
        """測試錯誤處理裝飾器 - 成功情況"""
        @handle_api_errors
        def test_func():
            return APIResponse.success(data={"result": "ok"})
        
        with app.app_context():
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 200
            assert data["status"] == "success"
    
    def test_handle_api_errors_validation_error(self, app):
        """測試錯誤處理裝飾器 - 驗證錯誤"""
        @handle_api_errors
        def test_func():
            raise ValidationError({"field": ["錯誤訊息"]})
        
        with app.app_context():
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 422
            assert data["error"]["code"] == "VALIDATION_ERROR"
    
    def test_handle_api_errors_value_error(self, app):
        """測試錯誤處理裝飾器 - 值錯誤"""
        @handle_api_errors
        def test_func():
            raise ValueError("無效的值")
        
        with app.app_context():
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 400
            assert data["error"]["code"] == "VALUE_ERROR"
            assert data["message"] == "無效的值"
    
    def test_handle_api_errors_key_error(self, app):
        """測試錯誤處理裝飾器 - 鍵錯誤"""
        @handle_api_errors
        def test_func():
            raise KeyError("missing_key")
        
        with app.app_context():
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 400
            assert data["error"]["code"] == "MISSING_PARAMETER"
    
    def test_handle_api_errors_permission_error(self, app):
        """測試錯誤處理裝飾器 - 權限錯誤"""
        @handle_api_errors
        def test_func():
            raise PermissionError("無權訪問")
        
        with app.app_context():
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 403
            assert data["message"] == "無權訪問"
    
    def test_handle_api_errors_file_not_found(self, app):
        """測試錯誤處理裝飾器 - 文件不存在"""
        @handle_api_errors
        def test_func():
            raise FileNotFoundError("檔案不存在")
        
        with app.app_context():
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 404
            assert data["message"] == "檔案不存在"
    
    @patch('utils.decorators.handle_exception')
    def test_handle_api_errors_generic_exception(self, mock_handle, app):
        """測試錯誤處理裝飾器 - 通用異常"""
        mock_handle.return_value = (Mock(), 500)
        
        @handle_api_errors
        def test_func():
            raise Exception("未知錯誤")
        
        with app.app_context():
            test_func()
            mock_handle.assert_called_once()
    
    def test_validate_request_json_success(self, app):
        """測試 JSON 請求驗證裝飾器 - 成功"""
        @validate_request_json(TestSchema)
        def test_func():
            return APIResponse.success(data=request.validated_data)
        
        with app.test_request_context(
            json={"name": "Test", "age": 25},
            content_type='application/json'
        ):
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 200
            assert data["data"]["name"] == "Test"
            assert data["data"]["age"] == 25
    
    def test_validate_request_json_not_json(self, app):
        """測試 JSON 請求驗證裝飾器 - 非 JSON"""
        @validate_request_json(TestSchema)
        def test_func():
            return APIResponse.success()
        
        with app.test_request_context(data="not json"):
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 400
            assert data["error"]["code"] == "INVALID_CONTENT_TYPE"
    
    def test_validate_request_json_validation_error(self, app):
        """測試 JSON 請求驗證裝飾器 - 驗證失敗"""
        @validate_request_json(TestSchema)
        def test_func():
            return APIResponse.success()
        
        with app.test_request_context(
            json={"name": "Test", "age": -1},
            content_type='application/json'
        ):
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 422
            assert "age" in data["error"]["details"]
    
    def test_validate_query_params_success(self, app):
        """測試查詢參數驗證裝飾器 - 成功"""
        @validate_query_params(TestSchema)
        def test_func():
            return APIResponse.success(data=request.validated_args)
        
        with app.test_request_context("/?name=Test&age=25"):
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 200
            assert data["data"]["name"] == "Test"
            assert data["data"]["age"] == 25
    
    def test_validate_query_params_validation_error(self, app):
        """測試查詢參數驗證裝飾器 - 驗證失敗"""
        @validate_query_params(TestSchema)
        def test_func():
            return APIResponse.success()
        
        with app.test_request_context("/?name=Test"):
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 422
            assert "age" in data["error"]["details"]
    
    @patch('utils.decorators.time.time')
    @patch('utils.decorators.current_app')
    def test_measure_performance(self, mock_app, mock_time, app):
        """測試性能測量裝飾器"""
        mock_time.side_effect = [0, 0.1]  # 100ms
        mock_logger = Mock()
        mock_app.logger = mock_logger
        
        @measure_performance
        def test_func():
            response = Mock()
            response.headers = {}
            return response, 200
        
        with app.app_context():
            response, status = test_func()
            
            assert response.headers['X-Execution-Time'] == "100.00ms"
            mock_logger.debug.assert_called_once()
    
    @patch('utils.decorators.time.time')
    @patch('utils.decorators.current_app')
    def test_measure_performance_error(self, mock_app, mock_time, app):
        """測試性能測量裝飾器 - 錯誤情況"""
        mock_time.side_effect = [0, 0.1]  # 100ms
        mock_logger = Mock()
        mock_app.logger = mock_logger
        
        @measure_performance
        def test_func():
            raise Exception("Test error")
        
        with app.app_context():
            with pytest.raises(Exception):
                test_func()
            
            mock_logger.error.assert_called_once()
    
    def test_require_json_content_type_success(self, app):
        """測試要求 JSON Content-Type 裝飾器 - 成功"""
        @require_json_content_type
        def test_func():
            return APIResponse.success()
        
        with app.test_request_context(
            headers={'Content-Type': 'application/json'}
        ):
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 200
    
    def test_require_json_content_type_missing(self, app):
        """測試要求 JSON Content-Type 裝飾器 - 缺少"""
        @require_json_content_type
        def test_func():
            return APIResponse.success()
        
        with app.test_request_context(
            headers={'Content-Type': 'text/plain'}
        ):
            response, status = test_func()
            data = json.loads(response.data)
            
            assert status == 400
            assert data["error"]["code"] == "INVALID_CONTENT_TYPE"
    
    @patch('utils.decorators.current_app')
    def test_log_request_response(self, mock_app, app):
        """測試請求響應日誌裝飾器"""
        mock_logger = Mock()
        mock_app.logger = mock_logger
        
        @log_request_response
        def test_func():
            return Mock(), 200
        
        with app.test_request_context(
            "/?param=value",
            method='GET',
            json={"test": "data"}
        ):
            test_func()
            
            # 應該記錄請求和響應
            assert mock_logger.info.call_count == 2