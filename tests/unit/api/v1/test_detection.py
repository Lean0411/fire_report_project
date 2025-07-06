"""
測試 detection API v1 端點
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from flask import Flask
from io import BytesIO
from api.v1.detection import detection_bp


@pytest.fixture
def app():
    """創建測試應用"""
    app = Flask(__name__)
    app.register_blueprint(detection_bp)
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = '/tmp/test_uploads'
    return app


@pytest.fixture
def client(app):
    """創建測試客戶端"""
    return app.test_client()


@pytest.fixture
def mock_image_file():
    """創建模擬圖片檔案"""
    file_data = BytesIO(b'fake image data')
    file_data.name = 'test.jpg'
    return file_data


class TestDetectionAPIV1:
    """Detection API v1 測試類"""
    
    @patch('api.v1.detection.detection_repo')
    @patch('api.v1.detection.secure_save_file')
    @patch('api.v1.detection.model_manager')
    @patch('api.v1.detection.ai_service')
    @patch('api.v1.detection.image_service')
    def test_analyze_image_success(self, mock_image_service, mock_ai_service, 
                                  mock_model_manager, mock_secure_save, 
                                  mock_detection_repo, client, mock_image_file):
        """測試成功分析圖片"""
        # 準備測試數據
        mock_secure_save.return_value = '/tmp/test.jpg'
        mock_model_manager.predict.return_value = {
            'is_fire': True,
            'confidence': 0.95,
            'probabilities': {'fire': 0.95, 'no_fire': 0.05}
        }
        mock_ai_service.generate_smart_advice.return_value = '請立即撤離並撥打119'
        mock_image_service.generate_annotated_image.return_value = '/tmp/annotated.jpg'
        
        mock_history = MagicMock()
        mock_history.id = 1
        mock_detection_repo.create_detection.return_value = mock_history
        
        # 創建模擬用戶
        mock_user = MagicMock()
        mock_user.id = 1
        
        # 發送請求
        with patch('api.v1.detection.request') as mock_request:
            mock_request.current_user = mock_user
            response = client.post('/api/v1/detection/analyze',
                                 data={
                                     'image': (mock_image_file, 'test.jpg'),
                                     'save_history': 'true'
                                 },
                                 content_type='multipart/form-data')
        
        # 驗證結果
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['detection']['is_fire'] is True
        assert data['detection']['confidence'] == 0.95
        assert 'ai_advice' in data['detection']
        
    def test_analyze_image_no_file(self, client):
        """測試沒有上傳檔案"""
        response = client.post('/api/v1/detection/analyze',
                             data={},
                             content_type='multipart/form-data')
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['error'] == 'No image file provided'
        
    @patch('api.v1.detection.detection_repo')
    def test_get_detection_history_success(self, mock_detection_repo, client):
        """測試成功獲取檢測歷史"""
        # 準備測試數據
        mock_history = MagicMock()
        mock_history.to_dict.return_value = {
            'id': 1,
            'is_fire': True,
            'confidence': 0.85,
            'created_at': '2024-01-01T00:00:00'
        }
        mock_detection_repo.get_user_detections.return_value = [mock_history]
        mock_detection_repo.count_user_detections.return_value = 1
        
        # 創建模擬用戶
        mock_user = MagicMock()
        mock_user.id = 1
        
        # 發送請求
        with patch('api.v1.detection.request') as mock_request:
            mock_request.current_user = mock_user
            response = client.get('/api/v1/detection/history')
        
        # 驗證結果
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert len(data['history']) == 1
        assert data['total'] == 1
        
    @patch('api.v1.detection.detection_repo')
    def test_get_detection_by_id_success(self, mock_detection_repo, client):
        """測試成功獲取單個檢測記錄"""
        # 準備測試數據
        mock_detection = MagicMock()
        mock_detection.user_id = 1
        mock_detection.to_dict.return_value = {
            'id': 1,
            'is_fire': True,
            'confidence': 0.9,
            'ai_advice': '火災警報'
        }
        mock_detection_repo.get_by_id.return_value = mock_detection
        
        # 創建模擬用戶
        mock_user = MagicMock()
        mock_user.id = 1
        
        # 發送請求
        with patch('api.v1.detection.request') as mock_request:
            mock_request.current_user = mock_user
            response = client.get('/api/v1/detection/1')
        
        # 驗證結果
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['detection']['id'] == 1
        
    def test_get_detection_by_id_not_found(self, client):
        """測試獲取不存在的檢測記錄"""
        with patch('api.v1.detection.detection_repo.get_by_id', return_value=None):
            response = client.get('/api/v1/detection/999')
            
            assert response.status_code == 404
            data = response.get_json()
            assert data['error'] == 'Detection not found'
            
    @patch('api.v1.detection.detection_repo')
    def test_delete_detection_success(self, mock_detection_repo, client):
        """測試成功刪除檢測記錄"""
        # 準備測試數據
        mock_detection = MagicMock()
        mock_detection.user_id = 1
        mock_detection_repo.get_by_id.return_value = mock_detection
        mock_detection_repo.delete.return_value = True
        
        # 創建模擬用戶
        mock_user = MagicMock()
        mock_user.id = 1
        
        # 發送請求
        with patch('api.v1.detection.request') as mock_request:
            mock_request.current_user = mock_user
            response = client.delete('/api/v1/detection/1')
        
        # 驗證結果
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['message'] == 'Detection deleted successfully'