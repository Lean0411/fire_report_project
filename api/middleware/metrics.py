from functools import wraps
from flask import request, current_app
import time
from datetime import datetime
from typing import Callable

from data.models.system_metrics import SystemMetrics, ApiUsage

def track_api_metrics(f: Callable) -> Callable:
    """Decorator to track API metrics"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # Get request information
        endpoint = request.endpoint or request.path
        method = request.method
        user_id = None
        api_key = None
        
        # Get user information if available
        if hasattr(request, 'current_user') and request.current_user:
            user_id = request.current_user.id
        
        # Get API key if present
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
        if api_key and api_key.startswith('Bearer '):
            api_key = api_key[7:]
        
        # Get request size
        request_size = request.content_length or 0
        
        try:
            # Execute the function
            response = f(*args, **kwargs)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Determine response status and size
            if isinstance(response, tuple):
                response_obj = response[0]
                status_code = response[1] if len(response) > 1 else 200
            else:
                response_obj = response
                status_code = 200
            
            # Calculate response size (approximate)
            response_size = 0
            if hasattr(response_obj, 'data'):
                response_size = len(response_obj.data)
            elif hasattr(response_obj, 'get_data'):
                response_size = len(response_obj.get_data())
            
            # Record API usage
            try:
                ApiUsage.record_api_call(
                    endpoint=endpoint,
                    method=method,
                    response_status=status_code,
                    response_time=response_time,
                    user_id=user_id,
                    api_key=api_key[:10] + '...' if api_key and len(api_key) > 10 else api_key,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', '')[:500],
                    request_size=request_size,
                    response_size=response_size
                )
                
                # Record performance metrics
                SystemMetrics.record_api_response_time(
                    endpoint=endpoint,
                    response_time=response_time,
                    user_id=user_id,
                    session_id=request.headers.get('X-Session-ID')
                )
                
            except Exception as e:
                current_app.logger.error(f"Failed to record API metrics: {str(e)}")
            
            return response
            
        except Exception as e:
            # Record error metrics
            response_time = (time.time() - start_time) * 1000
            
            try:
                # Record error in metrics
                SystemMetrics.record_error(
                    error_type=type(e).__name__,
                    endpoint=endpoint,
                    user_id=user_id,
                    error_details=str(e)[:500]
                )
                
                # Record failed API call
                ApiUsage.record_api_call(
                    endpoint=endpoint,
                    method=method,
                    response_status=500,
                    response_time=response_time,
                    user_id=user_id,
                    api_key=api_key[:10] + '...' if api_key and len(api_key) > 10 else api_key,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', '')[:500],
                    request_size=request_size,
                    response_size=0
                )
                
            except Exception as metric_error:
                current_app.logger.error(f"Failed to record error metrics: {str(metric_error)}")
            
            # Re-raise the original exception
            raise e
    
    return decorated_function

def track_model_performance(model_name: str):
    """Decorator to track model inference performance"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                
                # Calculate inference time
                inference_time = (time.time() - start_time) * 1000
                
                # Get user ID if available
                user_id = None
                if hasattr(request, 'current_user') and request.current_user:
                    user_id = request.current_user.id
                
                # Record model performance
                SystemMetrics.record_model_inference_time(
                    model_name=model_name,
                    inference_time=inference_time,
                    user_id=user_id
                )
                
                return result
                
            except Exception as e:
                # Record model error
                SystemMetrics.record_error(
                    error_type=f"ModelError_{type(e).__name__}",
                    user_id=getattr(request, 'current_user', {}).get('id') if hasattr(request, 'current_user') else None,
                    error_details=f"Model: {model_name}, Error: {str(e)}"[:500]
                )
                raise e
        
        return decorated_function
    return decorator