"""
測試 User model
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
from werkzeug.security import check_password_hash
from data.models.user_model import User


class TestUserModel:
    """User model 測試類"""
    
    @pytest.fixture
    def user(self):
        """創建測試用戶"""
        return User('testuser', 'test@example.com', 'Password123!')
    
    def test_init(self, user):
        """測試初始化"""
        assert user.username == 'testuser'
        assert user.email == 'test@example.com'
        assert user.role == 'user'
        assert user.password_hash is not None
        assert user.api_key is not None
    
    def test_init_with_role(self):
        """測試帶角色的初始化"""
        admin_user = User('admin', 'admin@example.com', 'Admin123!', role='admin')
        assert admin_user.role == 'admin'
    
    def test_set_password(self, user):
        """測試設置密碼"""
        user.set_password('NewPassword456!')
        
        # 驗證密碼哈希已更新
        assert user.password_hash is not None
        assert check_password_hash(user.password_hash, 'NewPassword456!')
    
    def test_check_password_correct(self, user):
        """測試驗證正確密碼"""
        user.set_password('TestPassword789!')
        
        assert user.check_password('TestPassword789!') is True
    
    def test_check_password_incorrect(self, user):
        """測試驗證錯誤密碼"""
        assert user.check_password('WrongPassword') is False
    
    def test_generate_api_key(self, user):
        """測試生成 API Key"""
        old_key = user.api_key
        new_key = user.generate_api_key()
        
        assert new_key != old_key
        assert user.api_key == new_key
        assert len(new_key) > 20  # 確保足夠長
    
    @patch('data.models.user_model.create_access_token')
    def test_get_access_token_default_expiry(self, mock_create_token, user):
        """測試獲取訪問令牌（預設過期時間）"""
        mock_create_token.return_value = 'fake-token'
        
        token = user.get_access_token()
        
        assert token == 'fake-token'
        mock_create_token.assert_called_once()
        call_args = mock_create_token.call_args
        assert call_args[1]['identity'] == user.id
        assert call_args[1]['expires_delta'] == timedelta(hours=24)
    
    @patch('data.models.user_model.create_access_token')
    def test_get_access_token_custom_expiry(self, mock_create_token, user):
        """測試獲取訪問令牌（自定義過期時間）"""
        mock_create_token.return_value = 'fake-refresh-token'
        custom_delta = timedelta(days=7)
        
        token = user.get_access_token(expires_delta=custom_delta)
        
        assert token == 'fake-refresh-token'
        call_args = mock_create_token.call_args
        assert call_args[1]['expires_delta'] == custom_delta
    
    @patch('data.models.user_model.db.session')
    def test_update_last_login(self, mock_session, user):
        """測試更新最後登入時間"""
        before_update = user.last_login
        
        with patch('data.models.user_model.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            user.update_last_login()
            
            assert user.last_login == mock_now
            mock_session.commit.assert_called_once()
    
    def test_has_role(self, user):
        """測試角色檢查"""
        assert user.has_role('user') is True
        assert user.has_role('admin') is False
        assert user.has_role('firefighter') is False
    
    def test_is_admin(self, user):
        """測試是否為管理員"""
        assert user.is_admin() is False
        
        user.role = 'admin'
        assert user.is_admin() is True
    
    def test_is_firefighter(self, user):
        """測試是否為消防員"""
        assert user.is_firefighter() is False
        
        user.role = 'firefighter'
        assert user.is_firefighter() is True
    
    def test_to_dict_without_sensitive(self, user):
        """測試轉換為字典（不含敏感資訊）"""
        user.id = 1
        user.last_login = datetime(2024, 1, 1, 10, 0, 0)
        user.created_at = datetime(2024, 1, 1, 8, 0, 0)
        user.updated_at = datetime(2024, 1, 1, 9, 0, 0)
        
        result = user.to_dict()
        
        assert result['id'] == 1
        assert result['username'] == 'testuser'
        assert result['email'] == 'test@example.com'
        assert result['role'] == 'user'
        assert result['is_active'] is True
        assert 'api_key' not in result
        assert 'password_hash' not in result
    
    def test_to_dict_with_sensitive(self, user):
        """測試轉換為字典（含敏感資訊）"""
        user.api_key = 'test-api-key-12345'
        
        result = user.to_dict(include_sensitive=True)
        
        assert 'api_key' in result
        assert result['api_key'] == 'test-api-key-12345'
    
    def test_repr(self, user):
        """測試字串表示"""
        assert repr(user) == '<User testuser>'
    
    def test_password_hash_security(self, user):
        """測試密碼哈希安全性"""
        # 確保密碼不是明文儲存
        assert 'Password123!' not in user.password_hash
        
        # 確保相同密碼產生不同哈希（加鹽）
        user2 = User('user2', 'user2@example.com', 'Password123!')
        assert user.password_hash != user2.password_hash
    
    def test_api_key_uniqueness(self):
        """測試 API Key 唯一性"""
        users = [User(f'user{i}', f'user{i}@example.com', 'Password123!') for i in range(10)]
        api_keys = [user.api_key for user in users]
        
        # 確保所有 API Key 都是唯一的
        assert len(api_keys) == len(set(api_keys))
    
    def test_relationships(self, user):
        """測試關聯關係"""
        # 確保 detection_history 關聯存在
        assert hasattr(user, 'detection_history')
        
    def test_table_name(self):
        """測試表名"""
        assert User.__tablename__ == 'users'