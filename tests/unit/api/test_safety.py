"""
測試 safety API 端點
"""
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from api.safety import safety_bp


@pytest.fixture
def app():
    """創建測試應用"""
    app = Flask(__name__)
    app.register_blueprint(safety_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """創建測試客戶端"""
    return app.test_client()


class TestSafetyAPI:
    """Safety API 測試類"""
    
    @patch('api.safety.sop_service')
    def test_get_safety_suggestions_success(self, mock_sop_service, client):
        """測試成功獲取安全建議"""
        # 準備測試數據
        mock_sop_service.get_relevant_sop.return_value = {
            'title': '火災應急程序',
            'content': '1. 保持冷靜\n2. 立即撥打119\n3. 使用滅火器',
            'category': 'general'
        }
        
        # 發送請求
        response = client.post('/api/safety', json={
            'is_fire': True,
            'role': 'general'
        })
        
        # 驗證結果
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'sop' in data
        assert data['sop']['title'] == '火災應急程序'
        
    def test_get_safety_suggestions_missing_params(self, client):
        """測試缺少參數的情況"""
        response = client.post('/api/safety', json={})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data
        
    def test_get_safety_suggestions_invalid_role(self, client):
        """測試無效角色"""
        response = client.post('/api/safety', json={
            'is_fire': True,
            'role': 'invalid_role'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data
        
    @patch('api.safety.sop_service')
    def test_get_safety_suggestions_no_sop_found(self, mock_sop_service, client):
        """測試找不到 SOP 的情況"""
        mock_sop_service.get_relevant_sop.return_value = None
        
        response = client.post('/api/safety', json={
            'is_fire': False,
            'role': 'firefighter'
        })
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False
        assert '未找到相關的安全建議' in data['error']
        
    @patch('api.safety.sop_service')
    def test_get_safety_suggestions_service_error(self, mock_sop_service, client):
        """測試服務錯誤"""
        mock_sop_service.get_relevant_sop.side_effect = Exception('Service error')
        
        response = client.post('/api/safety', json={
            'is_fire': True,
            'role': 'management'
        })
        
        assert response.status_code == 500
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data