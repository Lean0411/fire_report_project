"""
測試 auth API v1 端點
"""
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_jwt_extended import JWTManager
from api.v1.auth import auth_bp


@pytest.fixture
def app():
    """創建測試應用"""
    app = Flask(__name__)
    app.config['JWT_SECRET_KEY'] = 'test-secret-key'
    app.config['TESTING'] = True
    jwt = JWTManager(app)
    app.register_blueprint(auth_bp)
    return app


@pytest.fixture
def client(app):
    """創建測試客戶端"""
    return app.test_client()


class TestAuthAPI:
    """Auth API 測試類"""
    
    @patch('api.v1.auth.auth_service')
    def test_register_success(self, mock_auth_service, client):
        """測試成功註冊"""
        # 準備測試數據
        mock_user = MagicMock()
        mock_user.to_dict.return_value = {
            'id': 1,
            'username': 'testuser',
            'email': 'test@example.com',
            'role': 'user'
        }
        mock_auth_service.create_user.return_value = mock_user
        
        # 發送請求
        response = client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'Test123!@#'
        })
        
        # 驗證結果
        assert response.status_code == 201
        data = response.get_json()
        assert data['success'] is True
        assert data['user']['username'] == 'testuser'
        
    def test_register_missing_fields(self, client):
        """測試註冊缺少必要欄位"""
        response = client.post('/api/v1/auth/register', json={
            'username': 'testuser'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        
    def test_register_weak_password(self, client):
        """測試弱密碼"""
        response = client.post('/api/v1/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': '123456'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert '密碼強度不足' in data['error']
        
    @patch('api.v1.auth.auth_service')
    def test_login_success(self, mock_auth_service, client):
        """測試成功登入"""
        # 準備測試數據
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = 'testuser'
        mock_user.update_last_login = MagicMock()
        mock_auth_service.authenticate_user.return_value = mock_user
        mock_auth_service.generate_tokens.return_value = {
            'access_token': 'fake-access-token',
            'refresh_token': 'fake-refresh-token',
            'token_type': 'Bearer'
        }
        
        # 發送請求
        response = client.post('/api/v1/auth/login', json={
            'username_or_email': 'testuser',
            'password': 'Test123!@#'
        })
        
        # 驗證結果
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'access_token' in data
        assert data['token_type'] == 'Bearer'
        
    def test_login_invalid_credentials(self, client):
        """測試無效憑證登入"""
        with patch('api.v1.auth.auth_service.authenticate_user', return_value=None):
            response = client.post('/api/v1/auth/login', json={
                'username_or_email': 'testuser',
                'password': 'wrongpassword'
            })
            
            assert response.status_code == 401
            data = response.get_json()
            assert data['error'] == 'Invalid credentials'
            
    @patch('api.v1.auth.auth_service')
    def test_logout_success(self, mock_auth_service, client):
        """測試成功登出"""
        # 模擬已登入狀態
        with patch('api.v1.auth.get_jwt_identity', return_value=1):
            response = client.post('/api/v1/auth/logout')
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert data['message'] == 'Successfully logged out'
            
    @patch('api.v1.auth.auth_service')
    def test_get_current_user(self, mock_auth_service, client):
        """測試獲取當前用戶資訊"""
        # 準備測試數據
        mock_user = MagicMock()
        mock_user.to_dict.return_value = {
            'id': 1,
            'username': 'testuser',
            'email': 'test@example.com',
            'role': 'user'
        }
        
        # 模擬已登入狀態
        with patch('api.v1.auth.request') as mock_request:
            mock_request.current_user = mock_user
            response = client.get('/api/v1/auth/me')
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert data['user']['username'] == 'testuser'