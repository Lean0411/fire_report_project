"""
實用裝飾器集合
"""
from functools import wraps
from typing import Callable, Any
from flask import request, current_app
from marshmallow import ValidationError
from utils.api_response import APIResponse, handle_exception, format_validation_errors
import time


def handle_api_errors(func: Callable) -> Callable:
    """
    API 錯誤處理裝飾器
    自動捕獲異常並返回統一格式的錯誤響應
    
    Args:
        func: 被裝飾的函數
        
    Returns:
        包裝後的函數
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            # Marshmallow 驗證錯誤
            return APIResponse.validation_error(
                errors=format_validation_errors(e.messages)
            )
        except ValueError as e:
            # 值錯誤
            return APIResponse.error(
                message=str(e),
                error_code="VALUE_ERROR",
                status_code=400
            )
        except KeyError as e:
            # 鍵錯誤
            return APIResponse.error(
                message=f"缺少必要參數: {str(e)}",
                error_code="MISSING_PARAMETER",
                status_code=400
            )
        except PermissionError as e:
            # 權限錯誤
            return APIResponse.forbidden(
                message=str(e) or "無權執行此操作"
            )
        except FileNotFoundError as e:
            # 文件不存在
            return APIResponse.not_found(
                message=str(e) or "文件不存在"
            )
        except Exception as e:
            # 其他未處理的異常
            return handle_exception(e)
    
    return wrapper


def validate_request_json(schema_class):
    """
    請求 JSON 驗證裝飾器
    
    Args:
        schema_class: Marshmallow Schema 類
        
    Returns:
        裝飾器函數
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return APIResponse.error(
                    message="請求必須是 JSON 格式",
                    error_code="INVALID_CONTENT_TYPE",
                    status_code=400
                )
            
            schema = schema_class()
            try:
                data = schema.load(request.get_json())
                request.validated_data = data
                return func(*args, **kwargs)
            except ValidationError as e:
                return APIResponse.validation_error(
                    errors=format_validation_errors(e.messages)
                )
        
        return wrapper
    return decorator


def validate_query_params(schema_class):
    """
    查詢參數驗證裝飾器
    
    Args:
        schema_class: Marshmallow Schema 類
        
    Returns:
        裝飾器函數
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            schema = schema_class()
            try:
                data = schema.load(request.args.to_dict())
                request.validated_args = data
                return func(*args, **kwargs)
            except ValidationError as e:
                return APIResponse.validation_error(
                    errors=format_validation_errors(e.messages)
                )
        
        return wrapper
    return decorator


def measure_performance(func: Callable) -> Callable:
    """
    性能測量裝飾器
    記錄函數執行時間
    
    Args:
        func: 被裝飾的函數
        
    Returns:
        包裝後的函數
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # 計算執行時間
            execution_time = (time.time() - start_time) * 1000  # 毫秒
            
            # 如果是 tuple (response, status_code)
            if isinstance(result, tuple) and len(result) == 2:
                response, status_code = result
                # 嘗試添加執行時間到響應頭
                if hasattr(response, 'headers'):
                    response.headers['X-Execution-Time'] = f"{execution_time:.2f}ms"
            
            # 記錄性能日誌
            current_app.logger.debug(
                f"{func.__name__} executed in {execution_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            current_app.logger.error(
                f"{func.__name__} failed after {execution_time:.2f}ms: {str(e)}"
            )
            raise
    
    return wrapper


def async_task(func: Callable) -> Callable:
    """
    異步任務裝飾器（佔位符）
    未來可以整合 Celery 或其他異步任務隊列
    
    Args:
        func: 被裝飾的函數
        
    Returns:
        包裝後的函數
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: 實現異步任務隊列
        # 目前直接執行
        return func(*args, **kwargs)
    
    return wrapper


def cache_result(ttl: int = 300):
    """
    結果緩存裝飾器
    
    Args:
        ttl: 緩存時間（秒），默認 5 分鐘
        
    Returns:
        裝飾器函數
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # TODO: 實現 Redis 緩存
            # 目前直接執行
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_json_content_type(func: Callable) -> Callable:
    """
    要求 JSON Content-Type 的裝飾器
    
    Args:
        func: 被裝飾的函數
        
    Returns:
        包裝後的函數
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        content_type = request.headers.get('Content-Type', '')
        
        if not content_type.startswith('application/json'):
            return APIResponse.error(
                message="Content-Type 必須是 application/json",
                error_code="INVALID_CONTENT_TYPE",
                status_code=400
            )
        
        return func(*args, **kwargs)
    
    return wrapper


def log_request_response(func: Callable) -> Callable:
    """
    記錄請求和響應的裝飾器
    
    Args:
        func: 被裝飾的函數
        
    Returns:
        包裝後的函數
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 記錄請求
        current_app.logger.info(
            f"Request: {request.method} {request.path} "
            f"- Args: {request.args.to_dict()} "
            f"- Body: {request.get_json() if request.is_json else 'Not JSON'}"
        )
        
        # 執行函數
        result = func(*args, **kwargs)
        
        # 記錄響應
        if isinstance(result, tuple) and len(result) == 2:
            response, status_code = result
            current_app.logger.info(
                f"Response: {status_code} - {request.path}"
            )
        
        return result
    
    return wrapper