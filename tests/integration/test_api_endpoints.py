"""
Integration tests for API endpoints.
"""
import pytest
import json
import io
from PIL import Image
import tempfile
import os
from unittest.mock import patch

# Skip all integration tests if Flask app is not available
try:
    import app
except ImportError:
    pytest.skip("Flask app not available for integration tests", allow_module_level=True)


@pytest.mark.integration
class TestFireDetectionAPI:
    """Integration tests for fire detection API."""
    
    def test_detect_endpoint_post_with_image(self, client, sample_fire_image):
        """Test POST to /detect endpoint with image file."""
        with open(sample_fire_image, 'rb') as img_file:
            data = {
                'file': (img_file, 'test_fire.jpg'),
                'role': 'general',
                'use_ai': 'false'
            }
            
            response = client.post(
                '/detect',
                data=data,
                content_type='multipart/form-data'
            )
            
            assert response.status_code in [200, 302]  # Success or redirect
    
    def test_detect_endpoint_missing_file(self, client):
        """Test POST to /detect endpoint without file."""
        data = {
            'role': 'general',
            'use_ai': 'false'
        }
        
        response = client.post('/detect', data=data)
        
        # Should return error or redirect with error
        assert response.status_code in [400, 302]
    
    def test_detect_endpoint_invalid_file_type(self, client):
        """Test POST to /detect endpoint with invalid file type."""
        # Create text file
        text_content = b"This is not an image"
        data = {
            'file': (io.BytesIO(text_content), 'test.txt'),
            'role': 'general'
        }
        
        response = client.post(
            '/detect',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Should reject non-image files
        assert response.status_code in [400, 302]
    
    @patch('services.ai_service.get_ai_response')
    def test_detect_endpoint_with_ai_analysis(self, mock_ai_service, client, sample_fire_image):
        """Test fire detection with AI analysis enabled."""
        mock_ai_service.return_value = {
            'analysis': 'Fire detected with high confidence',
            'recommendations': ['Evacuate immediately'],
            'risk_level': 'high'
        }
        
        with open(sample_fire_image, 'rb') as img_file:
            data = {
                'file': (img_file, 'test_fire.jpg'),
                'role': 'firefighter',
                'use_ai': 'true',
                'ai_provider': 'openai'
            }
            
            response = client.post(
                '/detect',
                data=data,
                content_type='multipart/form-data'
            )
            
            assert response.status_code in [200, 302]
    
    def test_detect_endpoint_different_roles(self, client, sample_fire_image):
        """Test detection endpoint with different user roles."""
        roles = ['general', 'firefighter', 'management']
        
        for role in roles:
            with open(sample_fire_image, 'rb') as img_file:
                data = {
                    'file': (img_file, 'test_fire.jpg'),
                    'role': role,
                    'use_ai': 'false'
                }
                
                response = client.post(
                    '/detect',
                    data=data,
                    content_type='multipart/form-data'
                )
                
                assert response.status_code in [200, 302]


@pytest.mark.integration
class TestSafetyAPI:
    """Integration tests for safety API endpoints."""
    
    def test_safety_advice_endpoint(self, client):
        """Test GET /safety/advice endpoint."""
        response = client.get('/safety/advice?role=general')
        
        assert response.status_code == 200
        # Should return JSON with safety advice
        if response.content_type == 'application/json':
            data = json.loads(response.data)
            assert 'advice' in data or 'recommendations' in data
    
    def test_safety_advice_different_roles(self, client):
        """Test safety advice for different roles."""
        roles = ['general', 'firefighter', 'management']
        
        for role in roles:
            response = client.get(f'/safety/advice?role={role}')
            assert response.status_code == 200
    
    def test_safety_advice_invalid_role(self, client):
        """Test safety advice with invalid role."""
        response = client.get('/safety/advice?role=invalid_role')
        
        # Should return error or default advice
        assert response.status_code in [200, 400]


@pytest.mark.integration
class TestStatusAPI:
    """Integration tests for status API endpoints."""
    
    def test_status_endpoint(self, client):
        """Test GET /status endpoint."""
        response = client.get('/status')
        
        assert response.status_code == 200
        
        if response.content_type == 'application/json':
            data = json.loads(response.data)
            assert 'status' in data
            assert data['status'] in ['healthy', 'ok', 'running']
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint if it exists."""
        endpoints_to_try = ['/health', '/healthcheck', '/ping']
        
        for endpoint in endpoints_to_try:
            response = client.get(endpoint)
            # Should either return 200 or 404 (if endpoint doesn't exist)
            assert response.status_code in [200, 404]


@pytest.mark.integration
class TestWebInterface:
    """Integration tests for web interface."""
    
    def test_home_page(self, client):
        """Test main page loads correctly."""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'FireGuard' in response.data or b'Fire Detection' in response.data
    
    def test_static_files(self, client):
        """Test static files are served correctly."""
        static_files = [
            '/static/css/custom.css',
            '/static/js/enhanced-effects.js',
            '/static/favicon.ico'
        ]
        
        for static_file in static_files:
            response = client.get(static_file)
            # Should return file or 404 if file doesn't exist
            assert response.status_code in [200, 404]
    
    def test_form_submission_workflow(self, client, sample_fire_image):
        """Test complete workflow from form to results."""
        # First, get the main page
        response = client.get('/')
        assert response.status_code == 200
        
        # Then submit a detection request
        with open(sample_fire_image, 'rb') as img_file:
            data = {
                'file': (img_file, 'test.jpg'),
                'role': 'general',
                'use_ai': 'false'
            }
            
            response = client.post(
                '/detect',
                data=data,
                content_type='multipart/form-data',
                follow_redirects=True  # Follow any redirects
            )
            
            assert response.status_code == 200


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_404_error_handling(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent-endpoint')
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        # Try GET on POST-only endpoint
        response = client.get('/detect')
        assert response.status_code in [405, 200, 302]  # Method not allowed or redirect
    
    def test_large_file_upload(self, client):
        """Test handling of oversized file uploads."""
        # Create a large dummy file
        large_content = b"x" * (20 * 1024 * 1024)  # 20MB
        
        data = {
            'file': (io.BytesIO(large_content), 'large_file.jpg'),
            'role': 'general'
        }
        
        response = client.post(
            '/detect',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Should handle large files gracefully (reject or process)
        assert response.status_code in [200, 302, 400, 413]


@pytest.mark.integration
@pytest.mark.slow
class TestDataFlow:
    """Integration tests for complete data flow."""
    
    @patch('models.model_utils.load_model')
    @patch('services.ai_service.get_ai_response')
    def test_complete_detection_flow(self, mock_ai_service, mock_load_model, client, sample_fire_image):
        """Test complete flow from image upload to AI analysis."""
        # Setup mocks
        from unittest.mock import Mock
        mock_model = Mock()
        mock_model.return_value = Mock(squeeze=Mock(return_value=Mock(item=Mock(return_value=0.87))))
        mock_load_model.return_value = mock_model
        
        mock_ai_service.return_value = {
            'analysis': 'Fire detected with high confidence',
            'recommendations': ['Call emergency services', 'Evacuate area'],
            'risk_level': 'high'
        }
        
        # Submit detection request
        with open(sample_fire_image, 'rb') as img_file:
            data = {
                'file': (img_file, 'integration_test.jpg'),
                'role': 'firefighter',
                'use_ai': 'true',
                'ai_provider': 'openai'
            }
            
            response = client.post(
                '/detect',
                data=data,
                content_type='multipart/form-data',
                follow_redirects=True
            )
            
            assert response.status_code == 200
            
            # Check that model was called
            mock_load_model.assert_called()
            
            # Check that AI service was called
            mock_ai_service.assert_called()
    
    def test_no_ai_detection_flow(self, client, sample_no_fire_image):
        """Test detection flow without AI analysis."""
        with open(sample_no_fire_image, 'rb') as img_file:
            data = {
                'file': (img_file, 'no_fire_test.jpg'),
                'role': 'general',
                'use_ai': 'false'
            }
            
            response = client.post(
                '/detect',
                data=data,
                content_type='multipart/form-data',
                follow_redirects=True
            )
            
            assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrency:
    """Integration tests for concurrent requests."""
    
    def test_multiple_simultaneous_requests(self, client, sample_fire_image):
        """Test handling multiple simultaneous detection requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            with open(sample_fire_image, 'rb') as img_file:
                data = {
                    'file': (img_file, f'concurrent_test_{threading.current_thread().ident}.jpg'),
                    'role': 'general',
                    'use_ai': 'false'
                }
                
                response = client.post(
                    '/detect',
                    data=data,
                    content_type='multipart/form-data'
                )
                
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(3):  # Test with 3 concurrent requests
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should complete successfully
        assert len(results) == 3
        for status_code in results:
            assert status_code in [200, 302]