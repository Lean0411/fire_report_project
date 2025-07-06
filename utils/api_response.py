"""
API 響應格式統一化工具
提供標準化的 API 響應格式
"""
from typing import Any, Dict, List, Optional, Union
from flask import jsonify, Response
from datetime import datetime
import traceback
from enum import Enum


class ResponseStatus(Enum):
    """響應狀態枚舉"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class APIResponse:
    """統一的 API 響應類"""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "操作成功",
        meta: Optional[Dict[str, Any]] = None,
        status_code: int = 200
    ) -> Response:
        """
        成功響應
        
        Args:
            data: 響應數據
            message: 成功訊息
            meta: 元數據（如分頁信息）
            status_code: HTTP 狀態碼
            
        Returns:
            Flask Response 對象
        """
        response = {
            "status": ResponseStatus.SUCCESS.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": data
        }
        
        if meta:
            response["meta"] = meta
            
        return jsonify(response), status_code
    
    @staticmethod
    def error(
        message: str = "操作失敗",
        errors: Optional[Union[Dict[str, Any], List[str]]] = None,
        error_code: Optional[str] = None,
        status_code: int = 400,
        trace_id: Optional[str] = None
    ) -> Response:
        """
        錯誤響應
        
        Args:
            message: 錯誤訊息
            errors: 詳細錯誤信息
            error_code: 錯誤代碼
            status_code: HTTP 狀態碼
            trace_id: 追蹤 ID
            
        Returns:
            Flask Response 對象
        """
        response = {
            "status": ResponseStatus.ERROR.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": {
                "code": error_code or f"ERR_{status_code}",
                "details": errors
            }
        }
        
        if trace_id:
            response["trace_id"] = trace_id
            
        return jsonify(response), status_code
    
    @staticmethod
    def paginated(
        data: List[Any],
        page: int,
        per_page: int,
        total: int,
        message: str = "查詢成功",
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Response:
        """
        分頁響應
        
        Args:
            data: 數據列表
            page: 當前頁碼
            per_page: 每頁數量
            total: 總數量
            message: 成功訊息
            extra_meta: 額外的元數據
            
        Returns:
            Flask Response 對象
        """
        total_pages = (total + per_page - 1) // per_page
        
        meta = {
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "has_prev": page > 1,
                "has_next": page < total_pages
            }
        }
        
        if extra_meta:
            meta.update(extra_meta)
            
        return APIResponse.success(
            data=data,
            message=message,
            meta=meta
        )
    
    @staticmethod
    def created(
        data: Any = None,
        message: str = "創建成功",
        location: Optional[str] = None
    ) -> Response:
        """
        創建成功響應
        
        Args:
            data: 創建的資源數據
            message: 成功訊息
            location: 資源位置 URL
            
        Returns:
            Flask Response 對象
        """
        response = APIResponse.success(data=data, message=message, status_code=201)
        
        if location:
            response[0].headers["Location"] = location
            
        return response
    
    @staticmethod
    def no_content(message: str = "操作成功") -> Response:
        """
        無內容響應（用於刪除等操作）
        
        Args:
            message: 成功訊息
            
        Returns:
            Flask Response 對象
        """
        return APIResponse.success(message=message, status_code=204)
    
    @staticmethod
    def validation_error(
        errors: Dict[str, List[str]],
        message: str = "驗證失敗"
    ) -> Response:
        """
        驗證錯誤響應
        
        Args:
            errors: 字段驗證錯誤
            message: 錯誤訊息
            
        Returns:
            Flask Response 對象
        """
        return APIResponse.error(
            message=message,
            errors=errors,
            error_code="VALIDATION_ERROR",
            status_code=422
        )
    
    @staticmethod
    def unauthorized(
        message: str = "未授權",
        error_code: str = "UNAUTHORIZED"
    ) -> Response:
        """
        未授權響應
        
        Args:
            message: 錯誤訊息
            error_code: 錯誤代碼
            
        Returns:
            Flask Response 對象
        """
        return APIResponse.error(
            message=message,
            error_code=error_code,
            status_code=401
        )
    
    @staticmethod
    def forbidden(
        message: str = "無權限",
        error_code: str = "FORBIDDEN"
    ) -> Response:
        """
        禁止訪問響應
        
        Args:
            message: 錯誤訊息
            error_code: 錯誤代碼
            
        Returns:
            Flask Response 對象
        """
        return APIResponse.error(
            message=message,
            error_code=error_code,
            status_code=403
        )
    
    @staticmethod
    def not_found(
        resource: str = "資源",
        message: Optional[str] = None
    ) -> Response:
        """
        資源不存在響應
        
        Args:
            resource: 資源名稱
            message: 自定義錯誤訊息
            
        Returns:
            Flask Response 對象
        """
        if not message:
            message = f"{resource}不存在"
            
        return APIResponse.error(
            message=message,
            error_code="NOT_FOUND",
            status_code=404
        )
    
    @staticmethod
    def method_not_allowed(
        allowed_methods: List[str],
        message: str = "方法不允許"
    ) -> Response:
        """
        方法不允許響應
        
        Args:
            allowed_methods: 允許的方法列表
            message: 錯誤訊息
            
        Returns:
            Flask Response 對象
        """
        return APIResponse.error(
            message=message,
            errors={"allowed_methods": allowed_methods},
            error_code="METHOD_NOT_ALLOWED",
            status_code=405
        )
    
    @staticmethod
    def conflict(
        message: str = "資源衝突",
        error_code: str = "CONFLICT"
    ) -> Response:
        """
        資源衝突響應
        
        Args:
            message: 錯誤訊息
            error_code: 錯誤代碼
            
        Returns:
            Flask Response 對象
        """
        return APIResponse.error(
            message=message,
            error_code=error_code,
            status_code=409
        )
    
    @staticmethod
    def server_error(
        message: str = "服務器內部錯誤",
        error_id: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None
    ) -> Response:
        """
        服務器錯誤響應
        
        Args:
            message: 錯誤訊息
            error_id: 錯誤 ID（用於追蹤）
            debug_info: 調試信息（僅在開發環境顯示）
            
        Returns:
            Flask Response 對象
        """
        from flask import current_app
        
        error_details = {}
        if error_id:
            error_details["error_id"] = error_id
            
        # 只在開發環境顯示調試信息
        if debug_info and current_app.debug:
            error_details["debug"] = debug_info
            
        return APIResponse.error(
            message=message,
            errors=error_details if error_details else None,
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500
        )
    
    @staticmethod
    def service_unavailable(
        message: str = "服務暫時不可用",
        retry_after: Optional[int] = None
    ) -> Response:
        """
        服務不可用響應
        
        Args:
            message: 錯誤訊息
            retry_after: 建議重試時間（秒）
            
        Returns:
            Flask Response 對象
        """
        response = APIResponse.error(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503
        )
        
        if retry_after:
            response[0].headers["Retry-After"] = str(retry_after)
            
        return response


def handle_exception(e: Exception) -> Response:
    """
    統一的異常處理器
    
    Args:
        e: 異常對象
        
    Returns:
        Flask Response 對象
    """
    from flask import current_app
    import uuid
    
    # 生成錯誤 ID
    error_id = str(uuid.uuid4())
    
    # 記錄錯誤
    current_app.logger.error(
        f"Unhandled exception (ID: {error_id}): {str(e)}",
        exc_info=True
    )
    
    # 調試信息
    debug_info = None
    if current_app.debug:
        debug_info = {
            "exception": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_tb(e.__traceback__)
        }
    
    return APIResponse.server_error(
        message="服務器發生錯誤，請稍後重試",
        error_id=error_id,
        debug_info=debug_info
    )


def format_validation_errors(errors: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    格式化驗證錯誤
    
    Args:
        errors: 原始錯誤字典
        
    Returns:
        格式化後的錯誤字典
    """
    formatted = {}
    
    for field, error in errors.items():
        if isinstance(error, list):
            formatted[field] = error
        elif isinstance(error, dict):
            # 處理嵌套錯誤
            for sub_field, sub_error in error.items():
                key = f"{field}.{sub_field}"
                formatted[key] = [str(sub_error)] if not isinstance(sub_error, list) else sub_error
        else:
            formatted[field] = [str(error)]
            
    return formatted