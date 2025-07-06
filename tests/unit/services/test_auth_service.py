"""
測試 AuthService
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
from services.auth.auth_service import AuthService


class TestAuthService:
    """AuthService 測試類"""
    
    @pytest.fixture
    def auth_service(self):
        """創建 AuthService 實例"""
        return AuthService()
    
    @pytest.fixture
    def mock_user(self):
        """創建模擬用戶"""
        user = MagicMock()
        user.id = 1
        user.username = 'testuser'
        user.email = 'test@example.com'
        user.role = 'user'
        user.is_active = True
        user.api_key = 'test-api-key'
        user.get_access_token.return_value = 'fake-access-token'
        return user
    
    def test_init(self, auth_service):
        """測試初始化"""
        assert auth_service.user_repo is not None
        assert auth_service.jwt is None
    
    @patch('services.auth.auth_service.JWTManager')
    def test_init_app(self, mock_jwt_manager, auth_service):
        """測試應用初始化"""
        app = MagicMock()
        app.config = {}
        
        auth_service.init_app(app)
        
        # 驗證 JWT 配置
        assert 'JWT_SECRET_KEY' in app.config
        assert 'JWT_ACCESS_TOKEN_EXPIRES' in app.config
        assert 'JWT_REFRESH_TOKEN_EXPIRES' in app.config
        mock_jwt_manager.assert_called_once_with(app)
    
    def test_authenticate_user_success(self, auth_service, mock_user):
        """測試成功認證用戶"""
        auth_service.user_repo.authenticate = MagicMock(return_value=mock_user)
        
        result = auth_service.authenticate_user('testuser', 'password123')
        
        assert result == mock_user
        auth_service.user_repo.authenticate.assert_called_once_with('testuser', 'password123')
    
    def test_authenticate_user_failure(self, auth_service):
        """測試認證失敗"""
        auth_service.user_repo.authenticate = MagicMock(return_value=None)
        
        result = auth_service.authenticate_user('testuser', 'wrongpassword')
        
        assert result is None
    
    def test_authenticate_api_key_success(self, auth_service, mock_user):
        """測試成功認證 API Key"""
        auth_service.user_repo.find_by_api_key = MagicMock(return_value=mock_user)
        
        result = auth_service.authenticate_api_key('test-api-key')
        
        assert result == mock_user
        auth_service.user_repo.find_by_api_key.assert_called_once_with('test-api-key')
    
    def test_create_user(self, auth_service, mock_user):
        """測試創建用戶"""
        auth_service.user_repo.create_user = MagicMock(return_value=mock_user)
        
        result = auth_service.create_user('testuser', 'test@example.com', 'password123')
        
        assert result == mock_user
        auth_service.user_repo.create_user.assert_called_once_with(
            'testuser', 'test@example.com', 'password123', 'user'
        )
    
    def test_generate_tokens(self, auth_service, mock_user):
        """測試生成 tokens"""
        mock_user.get_access_token.side_effect = [
            'access-token',
            'refresh-token'
        ]
        
        result = auth_service.generate_tokens(mock_user)
        
        assert result['access_token'] == 'access-token'
        assert result['refresh_token'] == 'refresh-token'
        assert result['token_type'] == 'Bearer'
    
    def test_verify_api_key_with_bearer(self, auth_service, mock_user):
        """測試驗證帶 Bearer 前綴的 API Key"""
        auth_service.user_repo.find_by_api_key = MagicMock(return_value=mock_user)
        
        result = auth_service.verify_api_key('Bearer test-api-key')
        
        assert result == mock_user
        auth_service.user_repo.find_by_api_key.assert_called_once_with('test-api-key')
    
    def test_verify_api_key_empty(self, auth_service):
        """測試空 API Key"""
        result = auth_service.verify_api_key('')
        assert result is None
    
    @patch('services.auth.auth_service.verify_jwt_in_request')
    @patch('services.auth.auth_service.get_jwt_identity')
    def test_get_current_user_success(self, mock_get_jwt_identity, 
                                     mock_verify_jwt, auth_service, mock_user):
        """測試成功獲取當前用戶"""
        mock_get_jwt_identity.return_value = 1
        auth_service.user_repo.get_by_id = MagicMock(return_value=mock_user)
        
        result = auth_service.get_current_user()
        
        assert result == mock_user
        mock_verify_jwt.assert_called_once()
        mock_get_jwt_identity.assert_called_once()
    
    @patch('services.auth.auth_service.verify_jwt_in_request')
    def test_get_current_user_no_jwt(self, mock_verify_jwt, auth_service):
        """測試沒有 JWT 的情況"""
        mock_verify_jwt.side_effect = Exception('No JWT')
        
        result = auth_service.get_current_user()
        
        assert result is None
    
    def test_require_auth_decorator_with_jwt(self, auth_service):
        """測試 require_auth 裝飾器 - JWT 認證"""
        # 創建測試函數
        @auth_service.require_auth
        def test_func():
            return 'success'
        
        # 模擬 JWT 驗證成功
        with patch('services.auth.auth_service.verify_jwt_in_request'):
            with patch('flask.request'):
                result = test_func()
                assert result == 'success'
    
    def test_require_auth_decorator_with_api_key(self, auth_service, mock_user):
        """測試 require_auth 裝飾器 - API Key 認證"""
        # 創建測試函數
        @auth_service.require_auth
        def test_func():
            return 'success'
        
        # 模擬 JWT 驗證失敗，但 API Key 驗證成功
        with patch('services.auth.auth_service.verify_jwt_in_request', side_effect=Exception()):
            with patch('flask.request') as mock_request:
                mock_request.headers = {'X-API-Key': 'test-api-key'}
                auth_service.verify_api_key = MagicMock(return_value=mock_user)
                
                result = test_func()
                assert result == 'success'
    
    def test_require_role_decorator_admin(self, auth_service, mock_user):
        """測試 require_role 裝飾器 - 管理員角色"""
        mock_user.is_admin.return_value = True
        
        decorator = auth_service.require_role('admin')
        
        @decorator
        def test_func():
            return 'admin access'
        
        with patch('services.auth.auth_service.get_jwt_identity', return_value=1):
            with patch('flask.request') as mock_request:
                auth_service.user_repo.get_by_id = MagicMock(return_value=mock_user)
                
                result = test_func()
                assert result == 'admin access'