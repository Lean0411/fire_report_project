from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime, timedelta
import time
from collections import defaultdict
from typing import Dict, Optional

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.api_key_requests = defaultdict(list)
        self.user_requests = defaultdict(list)
    
    def is_allowed(self, key: str, limit: int, window: int) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed
        
        Args:
            key: Unique identifier (IP, user_id, api_key)
            limit: Number of requests allowed
            window: Time window in seconds
            
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        cutoff = now - window
        
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > cutoff]
        
        if len(self.requests[key]) >= limit:
            # Calculate retry after time
            oldest_request = min(self.requests[key])
            retry_after = int(oldest_request + window - now)
            return False, max(retry_after, 1)
        
        # Add current request
        self.requests[key].append(now)
        return True, None
    
    def cleanup_old_requests(self, max_age: int = 3600):
        """Clean up old request records"""
        cutoff = time.time() - max_age
        
        for key in list(self.requests.keys()):
            self.requests[key] = [req_time for req_time in self.requests[key] if req_time > cutoff]
            if not self.requests[key]:
                del self.requests[key]

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(requests_per_minute: int = 60, per_user: bool = False):
    """
    Rate limiting decorator
    
    Args:
        requests_per_minute: Number of requests allowed per minute
        per_user: If True, limit per authenticated user, otherwise per IP
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Determine the key for rate limiting
            if per_user and hasattr(request, 'current_user') and request.current_user:
                key = f"user_{request.current_user.id}"
            elif request.headers.get('X-API-Key'):
                # Rate limit by API key
                api_key = request.headers.get('X-API-Key')
                key = f"api_{api_key}"
            else:
                # Rate limit by IP address
                key = f"ip_{request.remote_addr}"
            
            # Check rate limit
            is_allowed, retry_after = rate_limiter.is_allowed(
                key=key,
                limit=requests_per_minute,
                window=60  # 60 seconds window
            )
            
            if not is_allowed:
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': retry_after,
                    'limit': requests_per_minute,
                    'window': '1 minute'
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(retry_after)
                response.headers['X-RateLimit-Limit'] = str(requests_per_minute)
                response.headers['X-RateLimit-Remaining'] = '0'
                response.headers['X-RateLimit-Reset'] = str(int(time.time() + retry_after))
                return response
            
            # Add rate limit headers to successful responses
            remaining = requests_per_minute - len(rate_limiter.requests[key])
            
            response = f(*args, **kwargs)
            
            # If response is a tuple, extract the response object
            if isinstance(response, tuple):
                response_obj = response[0]
                status_code = response[1] if len(response) > 1 else 200
            else:
                response_obj = response
                status_code = 200
            
            # Add headers if response is a Flask response object
            if hasattr(response_obj, 'headers'):
                response_obj.headers['X-RateLimit-Limit'] = str(requests_per_minute)
                response_obj.headers['X-RateLimit-Remaining'] = str(max(0, remaining))
                response_obj.headers['X-RateLimit-Reset'] = str(int(time.time() + 60))
            
            return response
        
        return decorated_function
    return decorator

def api_rate_limit(requests_per_minute: int = 100):
    """Rate limit for API endpoints"""
    return rate_limit(requests_per_minute=requests_per_minute, per_user=True)

def strict_rate_limit(requests_per_minute: int = 10):
    """Strict rate limit for sensitive endpoints"""
    return rate_limit(requests_per_minute=requests_per_minute, per_user=True)

def public_rate_limit(requests_per_minute: int = 30):
    """Rate limit for public endpoints (by IP)"""
    return rate_limit(requests_per_minute=requests_per_minute, per_user=False)