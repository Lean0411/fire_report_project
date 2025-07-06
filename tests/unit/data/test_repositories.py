"""
測試 repositories
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
from werkzeug.security import generate_password_hash
from data.repositories.user_repository import UserRepository
from data.repositories.detection_repository import DetectionRepository
from data.repositories.base_repository import BaseRepository
from data.models.user_model import User
from data.models.detection_history import DetectionHistory


class TestBaseRepository:
    """BaseRepository 測試類"""
    
    @pytest.fixture
    def base_repo(self):
        """創建 BaseRepository 實例"""
        repo = BaseRepository()
        repo.model = MagicMock()  # 設置模擬模型
        return repo
    
    @patch('data.repositories.base_repository.db.session')
    def test_get_by_id_found(self, mock_session, base_repo):
        """測試根據 ID 獲取 - 找到記錄"""
        # 準備測試數據
        mock_result = MagicMock()
        base_repo.model.query.get.return_value = mock_result
        
        # 調用方法
        result = base_repo.get_by_id(1)
        
        # 驗證
        assert result == mock_result
        base_repo.model.query.get.assert_called_once_with(1)
    
    def test_get_by_id_not_found(self, base_repo):
        """測試根據 ID 獲取 - 未找到記錄"""
        base_repo.model.query.get.return_value = None
        
        result = base_repo.get_by_id(999)
        
        assert result is None
    
    @patch('data.repositories.base_repository.db.session')
    def test_create_success(self, mock_session, base_repo):
        """測試創建記錄"""
        # 準備測試數據
        mock_instance = MagicMock()
        base_repo.model.return_value = mock_instance
        
        # 調用方法
        result = base_repo.create(name='test', value=123)
        
        # 驗證
        base_repo.model.assert_called_once_with(name='test', value=123)
        mock_session.add.assert_called_once_with(mock_instance)
        mock_session.commit.assert_called_once()
        assert result == mock_instance
    
    @patch('data.repositories.base_repository.db.session')
    def test_update_success(self, mock_session, base_repo):
        """測試更新記錄"""
        # 準備測試數據
        mock_instance = MagicMock()
        
        # 調用方法
        result = base_repo.update(mock_instance, name='updated', value=456)
        
        # 驗證屬性更新
        assert mock_instance.name == 'updated'
        assert mock_instance.value == 456
        mock_session.commit.assert_called_once()
        assert result == mock_instance
    
    @patch('data.repositories.base_repository.db.session')
    def test_delete_success(self, mock_session, base_repo):
        """測試刪除記錄"""
        # 準備測試數據
        mock_instance = MagicMock()
        
        # 調用方法
        result = base_repo.delete(mock_instance)
        
        # 驗證
        mock_session.delete.assert_called_once_with(mock_instance)
        mock_session.commit.assert_called_once()
        assert result is True
    
    def test_get_all(self, base_repo):
        """測試獲取所有記錄"""
        # 準備測試數據
        mock_results = [MagicMock(), MagicMock()]
        base_repo.model.query.all.return_value = mock_results
        
        # 調用方法
        result = base_repo.get_all()
        
        # 驗證
        assert result == mock_results
        base_repo.model.query.all.assert_called_once()


class TestUserRepository:
    """UserRepository 測試類"""
    
    @pytest.fixture
    def user_repo(self):
        """創建 UserRepository 實例"""
        return UserRepository()
    
    @pytest.fixture
    def mock_user(self):
        """創建模擬用戶"""
        user = MagicMock(spec=User)
        user.id = 1
        user.username = 'testuser'
        user.email = 'test@example.com'
        user.password_hash = generate_password_hash('password123')
        user.is_active = True
        user.check_password = MagicMock(return_value=True)
        return user
    
    def test_find_by_username_found(self, user_repo, mock_user):
        """測試根據用戶名查找 - 找到"""
        with patch.object(User.query, 'filter_by') as mock_filter:
            mock_filter.return_value.first.return_value = mock_user
            
            result = user_repo.find_by_username('testuser')
            
            assert result == mock_user
            mock_filter.assert_called_once_with(username='testuser')
    
    def test_find_by_username_not_found(self, user_repo):
        """測試根據用戶名查找 - 未找到"""
        with patch.object(User.query, 'filter_by') as mock_filter:
            mock_filter.return_value.first.return_value = None
            
            result = user_repo.find_by_username('nonexistent')
            
            assert result is None
    
    def test_find_by_email_found(self, user_repo, mock_user):
        """測試根據郵箱查找 - 找到"""
        with patch.object(User.query, 'filter_by') as mock_filter:
            mock_filter.return_value.first.return_value = mock_user
            
            result = user_repo.find_by_email('test@example.com')
            
            assert result == mock_user
            mock_filter.assert_called_once_with(email='test@example.com')
    
    def test_find_by_api_key_found(self, user_repo, mock_user):
        """測試根據 API Key 查找 - 找到"""
        with patch.object(User.query, 'filter_by') as mock_filter:
            mock_filter.return_value.first.return_value = mock_user
            
            result = user_repo.find_by_api_key('test-api-key')
            
            assert result == mock_user
            mock_filter.assert_called_once_with(api_key='test-api-key')
    
    @patch('data.repositories.user_repository.db.session')
    def test_create_user_success(self, mock_session, user_repo):
        """測試創建用戶成功"""
        with patch('data.models.user_model.User') as MockUser:
            mock_user = MagicMock()
            MockUser.return_value = mock_user
            
            result = user_repo.create_user('newuser', 'new@example.com', 'password123', 'user')
            
            MockUser.assert_called_once_with('newuser', 'new@example.com', 'password123', 'user')
            mock_session.add.assert_called_once_with(mock_user)
            mock_session.commit.assert_called_once()
            assert result == mock_user
    
    def test_authenticate_username_success(self, user_repo, mock_user):
        """測試使用用戶名認證成功"""
        user_repo.find_by_username = MagicMock(return_value=mock_user)
        
        result = user_repo.authenticate('testuser', 'password123')
        
        assert result == mock_user
        mock_user.check_password.assert_called_once_with('password123')
    
    def test_authenticate_email_success(self, user_repo, mock_user):
        """測試使用郵箱認證成功"""
        user_repo.find_by_username = MagicMock(return_value=None)
        user_repo.find_by_email = MagicMock(return_value=mock_user)
        
        result = user_repo.authenticate('test@example.com', 'password123')
        
        assert result == mock_user
        mock_user.check_password.assert_called_once_with('password123')
    
    def test_authenticate_wrong_password(self, user_repo, mock_user):
        """測試密碼錯誤"""
        user_repo.find_by_username = MagicMock(return_value=mock_user)
        mock_user.check_password.return_value = False
        
        result = user_repo.authenticate('testuser', 'wrongpassword')
        
        assert result is None
    
    def test_authenticate_user_not_found(self, user_repo):
        """測試用戶不存在"""
        user_repo.find_by_username = MagicMock(return_value=None)
        user_repo.find_by_email = MagicMock(return_value=None)
        
        result = user_repo.authenticate('nonexistent', 'password')
        
        assert result is None


class TestDetectionRepository:
    """DetectionRepository 測試類"""
    
    @pytest.fixture
    def detection_repo(self):
        """創建 DetectionRepository 實例"""
        return DetectionRepository()
    
    @pytest.fixture
    def mock_detection(self):
        """創建模擬檢測記錄"""
        detection = MagicMock(spec=DetectionHistory)
        detection.id = 1
        detection.user_id = 1
        detection.is_fire = True
        detection.confidence = 0.95
        detection.created_at = datetime.utcnow()
        return detection
    
    @patch('data.repositories.detection_repository.db.session')
    def test_create_detection_success(self, mock_session, detection_repo):
        """測試創建檢測記錄"""
        with patch('data.models.detection_history.DetectionHistory') as MockDetection:
            mock_detection = MagicMock()
            MockDetection.return_value = mock_detection
            
            result = detection_repo.create_detection(
                user_id=1,
                image_path='/tmp/test.jpg',
                is_fire=True,
                confidence=0.95,
                ai_advice='Fire detected!'
            )
            
            MockDetection.assert_called_once()
            mock_session.add.assert_called_once_with(mock_detection)
            mock_session.commit.assert_called_once()
            assert result == mock_detection
    
    def test_get_user_detections_with_pagination(self, detection_repo):
        """測試獲取用戶檢測記錄（分頁）"""
        mock_detections = [MagicMock(), MagicMock()]
        
        with patch.object(DetectionHistory.query, 'filter_by') as mock_filter:
            mock_filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_detections
            
            result = detection_repo.get_user_detections(user_id=1, page=2, per_page=10)
            
            assert result == mock_detections
            mock_filter.assert_called_once_with(user_id=1)
    
    def test_get_user_detections_no_pagination(self, detection_repo):
        """測試獲取用戶所有檢測記錄"""
        mock_detections = [MagicMock(), MagicMock(), MagicMock()]
        
        with patch.object(DetectionHistory.query, 'filter_by') as mock_filter:
            mock_filter.return_value.order_by.return_value.all.return_value = mock_detections
            
            result = detection_repo.get_user_detections(user_id=1)
            
            assert result == mock_detections
    
    def test_count_user_detections(self, detection_repo):
        """測試計算用戶檢測記錄數量"""
        with patch.object(DetectionHistory.query, 'filter_by') as mock_filter:
            mock_filter.return_value.count.return_value = 42
            
            result = detection_repo.count_user_detections(user_id=1)
            
            assert result == 42
            mock_filter.assert_called_once_with(user_id=1)
    
    def test_get_recent_detections(self, detection_repo):
        """測試獲取最近的檢測記錄"""
        mock_detections = [MagicMock(), MagicMock()]
        
        with patch.object(DetectionHistory.query, 'order_by') as mock_order:
            mock_order.return_value.limit.return_value.all.return_value = mock_detections
            
            result = detection_repo.get_recent_detections(limit=2)
            
            assert result == mock_detections
            assert len(result) == 2
    
    @patch('data.repositories.detection_repository.db.session')
    def test_delete_user_detections(self, mock_session, detection_repo):
        """測試刪除用戶所有檢測記錄"""
        with patch.object(DetectionHistory.query, 'filter_by') as mock_filter:
            mock_filter.return_value.delete.return_value = 5
            
            result = detection_repo.delete_user_detections(user_id=1)
            
            assert result == 5
            mock_filter.assert_called_once_with(user_id=1)
            mock_session.commit.assert_called_once()