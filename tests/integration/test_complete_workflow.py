"""
Integration tests for complete workflow scenarios.
"""
import pytest
import os
import tempfile
import json
from PIL import Image
import numpy as np
from unittest.mock import patch, Mock


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @patch('models.model_utils.load_model')
    @patch('services.ai_service.get_ai_response')
    def test_fire_detection_workflow_with_ai(self, mock_ai_service, mock_load_model, client):
        """Test complete fire detection workflow with AI analysis."""
        # Setup mocks
        mock_model = Mock()
        mock_model.return_value = Mock(squeeze=Mock(return_value=Mock(item=Mock(return_value=0.92))))
        mock_load_model.return_value = mock_model
        
        mock_ai_service.return_value = {
            'analysis': 'High confidence fire detection. Immediate action required.',
            'recommendations': [
                'Evacuate the area immediately',
                'Call emergency services (119)',
                'Do not attempt to fight the fire',
                'Ensure all personnel are accounted for'
            ],
            'risk_level': 'critical'
        }
        
        # Create test fire image
        fire_image = self._create_fire_test_image()
        
        try:
            # Step 1: Access main page
            response = client.get('/')
            assert response.status_code == 200
            
            # Step 2: Submit fire detection request
            with open(fire_image, 'rb') as img_file:
                data = {
                    'file': (img_file, 'emergency_fire.jpg'),
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
                
                # Verify model and AI service were called
                mock_load_model.assert_called()
                mock_ai_service.assert_called()
                
                # Check response contains expected data
                response_text = response.data.decode('utf-8')
                assert '0.92' in response_text or '92%' in response_text  # Confidence
                assert 'critical' in response_text.lower() or 'high' in response_text.lower()
        
        finally:
            os.unlink(fire_image)
    
    def test_no_fire_detection_workflow(self, client):
        """Test workflow when no fire is detected."""
        # Create test normal image
        normal_image = self._create_normal_test_image()
        
        try:
            with open(normal_image, 'rb') as img_file:
                data = {
                    'file': (img_file, 'safe_environment.jpg'),
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
                
                # Response should indicate no fire detected
                response_text = response.data.decode('utf-8').lower()
                assert 'safe' in response_text or 'no fire' in response_text or 'normal' in response_text
        
        finally:
            os.unlink(normal_image)
    
    def test_safety_consultation_workflow(self, client):
        """Test safety advice consultation workflow."""
        roles = ['general', 'firefighter', 'management']
        
        for role in roles:
            # Step 1: Request safety advice
            response = client.get(f'/safety/advice?role={role}')
            assert response.status_code == 200
            
            # Should receive role-appropriate advice
            if response.content_type == 'application/json':
                data = json.loads(response.data)
                assert 'advice' in data or 'recommendations' in data
    
    def test_multi_image_analysis_workflow(self, client):
        """Test workflow with multiple image analyses."""
        # Create multiple test images
        images = [
            self._create_fire_test_image(),
            self._create_normal_test_image(),
            self._create_smoke_test_image()
        ]
        
        try:
            results = []
            
            for i, image_path in enumerate(images):
                with open(image_path, 'rb') as img_file:
                    data = {
                        'file': (img_file, f'test_image_{i}.jpg'),
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
                    results.append(response)
            
            # All analyses should complete successfully
            assert len(results) == 3
        
        finally:
            for image_path in images:
                try:
                    os.unlink(image_path)
                except OSError:
                    pass


class TestErrorRecoveryWorkflow:
    """Test workflows with error conditions and recovery."""
    
    def test_invalid_file_recovery(self, client):
        """Test recovery from invalid file upload."""
        # Try to upload text file as image
        text_content = b"This is not an image file"
        
        data = {
            'file': (tempfile.NamedTemporaryFile(delete=False), 'fake_image.txt'),
            'role': 'general'
        }
        
        response = client.post(
            '/detect',
            data=data,
            content_type='multipart/form-data',
            follow_redirects=True
        )
        
        # Should handle error gracefully
        assert response.status_code in [200, 400]
        
        # Should show error message to user
        if response.status_code == 200:
            response_text = response.data.decode('utf-8').lower()
            assert 'error' in response_text or 'invalid' in response_text
    
    @patch('models.model_utils.load_model')
    def test_model_error_recovery(self, mock_load_model, client):
        """Test recovery from model loading errors."""
        # Make model loading fail
        mock_load_model.side_effect = Exception("Model loading failed")
        
        normal_image = self._create_normal_test_image()
        
        try:
            with open(normal_image, 'rb') as img_file:
                data = {
                    'file': (img_file, 'test_error.jpg'),
                    'role': 'general',
                    'use_ai': 'false'
                }
                
                response = client.post(
                    '/detect',
                    data=data,
                    content_type='multipart/form-data',
                    follow_redirects=True
                )
                
                # Should handle error gracefully
                assert response.status_code in [200, 500]
                
                if response.status_code == 200:
                    response_text = response.data.decode('utf-8').lower()
                    assert 'error' in response_text or 'unavailable' in response_text
        
        finally:
            os.unlink(normal_image)
    
    @patch('services.ai_service.get_ai_response')
    def test_ai_service_error_recovery(self, mock_ai_service, client):
        """Test recovery from AI service errors."""
        # Make AI service fail
        mock_ai_service.side_effect = Exception("AI service unavailable")
        
        fire_image = self._create_fire_test_image()
        
        try:
            with open(fire_image, 'rb') as img_file:
                data = {
                    'file': (img_file, 'test_ai_error.jpg'),
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
                
                # Should still provide basic detection results
                assert response.status_code == 200
                
                # Should indicate AI analysis unavailable
                response_text = response.data.decode('utf-8').lower()
                assert 'detection' in response_text  # Basic detection should work
        
        finally:
            os.unlink(fire_image)


class TestPerformanceWorkflow:
    """Test performance-related workflows."""
    
    def test_large_image_processing(self, client):
        """Test processing of large images."""
        # Create large image
        large_image = self._create_large_test_image()
        
        try:
            with open(large_image, 'rb') as img_file:
                data = {
                    'file': (img_file, 'large_test_image.jpg'),
                    'role': 'general',
                    'use_ai': 'false'
                }
                
                response = client.post(
                    '/detect',
                    data=data,
                    content_type='multipart/form-data',
                    follow_redirects=True
                )
                
                # Should handle large images (resize/compress as needed)
                assert response.status_code in [200, 413]  # Success or payload too large
        
        finally:
            os.unlink(large_image)
    
    def test_rapid_successive_requests(self, client):
        """Test rapid successive detection requests."""
        normal_image = self._create_normal_test_image()
        
        try:
            responses = []
            
            for i in range(5):  # 5 rapid requests
                with open(normal_image, 'rb') as img_file:
                    data = {
                        'file': (img_file, f'rapid_test_{i}.jpg'),
                        'role': 'general',
                        'use_ai': 'false'
                    }
                    
                    response = client.post(
                        '/detect',
                        data=data,
                        content_type='multipart/form-data',
                        follow_redirects=True
                    )
                    
                    responses.append(response.status_code)
            
            # All requests should be handled properly
            for status_code in responses:
                assert status_code in [200, 429]  # Success or rate limited
        
        finally:
            os.unlink(normal_image)


class TestSecurityWorkflow:
    """Test security-related workflows."""
    
    def test_malicious_filename_handling(self, client):
        """Test handling of malicious filenames."""
        normal_image = self._create_normal_test_image()
        
        malicious_names = [
            '../../../etc/passwd.jpg',
            '..\\..\\windows\\system32\\config.jpg',
            '<script>alert("xss")</script>.jpg',
            'file; rm -rf /.jpg'
        ]
        
        try:
            for malicious_name in malicious_names:
                with open(normal_image, 'rb') as img_file:
                    data = {
                        'file': (img_file, malicious_name),
                        'role': 'general',
                        'use_ai': 'false'
                    }
                    
                    response = client.post(
                        '/detect',
                        data=data,
                        content_type='multipart/form-data',
                        follow_redirects=True
                    )
                    
                    # Should handle securely without crashing
                    assert response.status_code in [200, 400]
        
        finally:
            os.unlink(normal_image)
    
    def test_invalid_role_handling(self, client):
        """Test handling of invalid user roles."""
        normal_image = self._create_normal_test_image()
        
        invalid_roles = ['admin', 'root', 'superuser', '<script>', '../../etc']
        
        try:
            for invalid_role in invalid_roles:
                with open(normal_image, 'rb') as img_file:
                    data = {
                        'file': (img_file, 'security_test.jpg'),
                        'role': invalid_role,
                        'use_ai': 'false'
                    }
                    
                    response = client.post(
                        '/detect',
                        data=data,
                        content_type='multipart/form-data',
                        follow_redirects=True
                    )
                    
                    # Should reject invalid roles or default to safe role
                    assert response.status_code in [200, 400]
        
        finally:
            os.unlink(normal_image)
    
    # Helper methods for creating test images
    def _create_fire_test_image(self):
        """Create a test image simulating fire."""
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red
        img_array[:, :, 1] = 165  # Orange
        img_array[:, :, 2] = 0    # Blue
        
        img = Image.fromarray(img_array)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG')
        return temp_file.name
    
    def _create_normal_test_image(self):
        """Create a test image simulating normal environment."""
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 0] = 0    # Red
        img_array[:, :, 1] = 128  # Green
        img_array[:, :, 2] = 255  # Blue
        
        img = Image.fromarray(img_array)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG')
        return temp_file.name
    
    def _create_smoke_test_image(self):
        """Create a test image simulating smoke."""
        img_array = np.full((224, 224, 3), 128, dtype=np.uint8)  # Gray smoke
        
        img = Image.fromarray(img_array)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG')
        return temp_file.name
    
    def _create_large_test_image(self):
        """Create a large test image."""
        img_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG', quality=95)
        return temp_file.name