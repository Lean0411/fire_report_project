from functools import wraps
from flask import request, jsonify, current_app
from flask_jwt_extended import JWTManager, verify_jwt_in_request, get_jwt_identity, get_jwt
from werkzeug.security import check_password_hash
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config.constants import (
    JWT_ACCESS_TOKEN_EXPIRES, JWT_REFRESH_TOKEN_EXPIRES,
    AUTH_BEARER_PREFIX_LENGTH
)

from data.repositories.user_repository import UserRepository
from data.models.user_model import User

class AuthService:
    """Authentication and authorization service"""
    
    def __init__(self):
        self.user_repo = UserRepository()
        self.jwt = None
    
    def init_app(self, app):
        """Initialize authentication with Flask app"""
        # JWT Configuration
        app.config['JWT_SECRET_KEY'] = app.config.get('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = JWT_ACCESS_TOKEN_EXPIRES
        app.config['JWT_REFRESH_TOKEN_EXPIRES'] = JWT_REFRESH_TOKEN_EXPIRES
        
        self.jwt = JWTManager(app)
        
        # JWT event handlers
        @self.jwt.user_identity_loader
        def user_identity_lookup(user):
            if isinstance(user, User):
                return user.id
            return user
        
        @self.jwt.user_lookup_loader
        def user_lookup_callback(_jwt_header, jwt_data):
            identity = jwt_data["sub"]
            return self.user_repo.get_by_id(identity)
        
        @self.jwt.expired_token_loader
        def expired_token_callback(jwt_header, jwt_payload):
            return jsonify({'error': 'Token has expired'}), 401
        
        @self.jwt.invalid_token_loader
        def invalid_token_callback(error):
            return jsonify({'error': 'Invalid token'}), 401
        
        @self.jwt.unauthorized_loader
        def missing_token_callback(error):
            return jsonify({'error': 'Authorization token required'}), 401
    
    def authenticate_user(self, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password"""
        return self.user_repo.authenticate(username_or_email, password)
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user with API key"""
        return self.user_repo.find_by_api_key(api_key)
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> User:
        """Create a new user"""
        return self.user_repo.create_user(username, email, password, role)
    
    def generate_tokens(self, user: User) -> Dict[str, str]:
        """Generate access and refresh tokens for user"""
        access_token = user.get_access_token()
        
        # For refresh token, we'll use a longer expiry
        refresh_token = user.get_access_token(expires_delta=JWT_REFRESH_TOKEN_EXPIRES)
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer'
        }
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify API key and return user"""
        if not api_key:
            return None
        
        # Remove 'Bearer ' prefix if present
        if api_key.startswith('Bearer '):
            api_key = api_key[AUTH_BEARER_PREFIX_LENGTH:]
        
        return self.authenticate_api_key(api_key)
    
    def get_current_user(self) -> Optional[User]:
        """Get current authenticated user from JWT token"""
        try:
            verify_jwt_in_request()
            user_id = get_jwt_identity()
            return self.user_repo.get_by_id(user_id)
        except Exception:
            return None
    
    def require_auth(self, f):
        """Decorator to require authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check for JWT token first
            try:
                verify_jwt_in_request()
                return f(*args, **kwargs)
            except Exception:
                pass
            
            # Check for API key in headers
            api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
            if api_key:
                user = self.verify_api_key(api_key)
                if user and user.is_active:
                    # Store user in request context
                    request.current_user = user
                    return f(*args, **kwargs)
            
            return jsonify({'error': 'Authentication required'}), 401
        
        return decorated_function
    
    def require_role(self, required_role: str):
        """Decorator to require specific role"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # First check authentication
                user = self.get_current_user()
                
                # If no JWT user, check API key
                if not user:
                    api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
                    if api_key:
                        user = self.verify_api_key(api_key)
                
                if not user:
                    return jsonify({'error': 'Authentication required'}), 401
                
                if not user.is_active:
                    return jsonify({'error': 'Account is deactivated'}), 403
                
                # Check role authorization
                if required_role == 'admin' and not user.is_admin():
                    return jsonify({'error': 'Admin privileges required'}), 403
                
                if required_role == 'firefighter' and not (user.is_firefighter() or user.is_admin()):
                    return jsonify({'error': 'Firefighter or admin privileges required'}), 403
                
                # Store user in request context
                request.current_user = user
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def require_admin(self, f):
        """Decorator to require admin role"""
        return self.require_role('admin')(f)
    
    def require_firefighter(self, f):
        """Decorator to require firefighter role"""
        return self.require_role('firefighter')(f)
    
    def optional_auth(self, f):
        """Decorator for optional authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = self.get_current_user()
            
            # If no JWT user, check API key
            if not user:
                api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
                if api_key:
                    user = self.verify_api_key(api_key)
            
            # Store user in request context (can be None)
            request.current_user = user
            return f(*args, **kwargs)
        
        return decorated_function

# Global auth service instance
auth_service = AuthService()