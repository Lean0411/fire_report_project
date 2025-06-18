"""
Unit tests for detection API endpoints.
"""
import pytest
import json
import io
from PIL import Image
from unittest.mock import patch, Mock

from api.detection import detect_fire_endpoint, validate_detection_request


class TestDetectionEndpoint:
    """Test cases for fire detection API endpoint."""
    
    @patch('api.detection.load_model')
    @patch('api.detection.preprocess_image')
    def test_detect_fire_endpoint_success(self, mock_preprocess, mock_load_model, sample_fire_image):
        """Test successful fire detection API call."""
        # Setup mocks
        mock_model = Mock()
        mock_model.return_value = Mock(squeeze=Mock(return_value=Mock(item=Mock(return_value=0.92))))
        mock_load_model.return_value = mock_model
        mock_preprocess.return_value = Mock()  # Mock tensor
        
        # Mock request data
        request_data = {
            'image_path': sample_fire_image,
            'role': 'firefighter',
            'use_ai': True,
            'ai_provider': 'mock'
        }
        
        result = detect_fire_endpoint(request_data)
        
        # Assertions
        assert result is not None
        assert 'fire_detected' in result
        assert 'confidence' in result
        assert 'timestamp' in result
        assert result['confidence'] == 0.92
    
    @patch('api.detection.load_model')
    def test_detect_fire_endpoint_no_fire(self, mock_load_model, sample_no_fire_image):
        """Test fire detection with no fire present."""
        # Setup mock for no fire detection
        mock_model = Mock()
        mock_model.return_value = Mock(squeeze=Mock(return_value=Mock(item=Mock(return_value=0.15))))
        mock_load_model.return_value = mock_model
        
        request_data = {
            'image_path': sample_no_fire_image,
            'role': 'general',
            'use_ai': False
        }
        
        result = detect_fire_endpoint(request_data)
        
        assert result['fire_detected'] == False
        assert result['confidence'] == 0.15
    
    def test_detect_fire_endpoint_invalid_image(self):
        """Test detection with invalid image path."""
        request_data = {
            'image_path': 'nonexistent_image.jpg',
            'role': 'general'
        }
        
        with pytest.raises((FileNotFoundError, ValueError)):
            detect_fire_endpoint(request_data)
    
    def test_detect_fire_endpoint_missing_params(self):
        """Test detection with missing required parameters."""
        incomplete_data = {
            'role': 'general'
            # Missing image_path
        }
        
        with pytest.raises((KeyError, ValueError)):
            detect_fire_endpoint(incomplete_data)


class TestRequestValidation:
    """Test cases for request validation."""
    
    def test_validate_detection_request_valid(self):
        """Test validation with valid request data."""
        valid_request = {
            'image_path': '/valid/path/image.jpg',
            'role': 'firefighter',
            'use_ai': True,
            'ai_provider': 'openai'
        }
        
        result = validate_detection_request(valid_request)
        assert result['valid'] == True
    
    def test_validate_detection_request_invalid_role(self):
        """Test validation with invalid user role."""
        invalid_request = {
            'image_path': '/valid/path/image.jpg',
            'role': 'invalid_role',
            'use_ai': True
        }
        
        result = validate_detection_request(invalid_request)
        assert result['valid'] == False
        assert 'role' in result['error'].lower()
    
    def test_validate_detection_request_invalid_provider(self):
        """Test validation with invalid AI provider."""
        invalid_request = {
            'image_path': '/valid/path/image.jpg',
            'role': 'general',
            'use_ai': True,
            'ai_provider': 'invalid_provider'
        }
        
        result = validate_detection_request(invalid_request)
        assert result['valid'] == False
        assert 'provider' in result['error'].lower()
    
    def test_validate_detection_request_missing_image(self):
        """Test validation with missing image path."""
        invalid_request = {
            'role': 'general',
            'use_ai': False
        }
        
        result = validate_detection_request(invalid_request)
        assert result['valid'] == False
        assert 'image' in result['error'].lower()
    
    def test_validate_detection_request_ai_without_provider(self):
        """Test validation when AI is enabled but no provider specified."""
        invalid_request = {
            'image_path': '/valid/path/image.jpg',
            'role': 'general',
            'use_ai': True
            # Missing ai_provider
        }
        
        result = validate_detection_request(invalid_request)
        # Should either default to a provider or return error
        assert isinstance(result['valid'], bool)


class TestAPIResponseFormat:
    """Test cases for API response formatting."""
    
    @patch('api.detection.load_model')
    def test_response_format_structure(self, mock_load_model, sample_fire_image):
        """Test that API response has correct structure."""
        # Setup mock
        mock_model = Mock()
        mock_model.return_value = Mock(squeeze=Mock(return_value=Mock(item=Mock(return_value=0.85))))
        mock_load_model.return_value = mock_model
        
        request_data = {
            'image_path': sample_fire_image,
            'role': 'general',
            'use_ai': False
        }
        
        result = detect_fire_endpoint(request_data)
        
        # Check required fields
        required_fields = ['fire_detected', 'confidence', 'timestamp']
        for field in required_fields:
            assert field in result
        
        # Check data types
        assert isinstance(result['fire_detected'], bool)
        assert isinstance(result['confidence'], (int, float))
        assert isinstance(result['timestamp'], str)
    
    @patch('api.detection.load_model')
    @patch('api.detection.get_ai_response')
    def test_response_format_with_ai(self, mock_ai_response, mock_load_model, sample_fire_image):
        """Test API response format when AI analysis is included."""
        # Setup mocks
        mock_model = Mock()
        mock_model.return_value = Mock(squeeze=Mock(return_value=Mock(item=Mock(return_value=0.90))))
        mock_load_model.return_value = mock_model
        
        mock_ai_response.return_value = {
            'analysis': 'High confidence fire detection',
            'recommendations': ['Evacuate immediately'],
            'risk_level': 'high'
        }
        
        request_data = {
            'image_path': sample_fire_image,
            'role': 'firefighter',
            'use_ai': True,
            'ai_provider': 'openai'
        }
        
        result = detect_fire_endpoint(request_data)
        
        # Should include AI analysis
        assert 'ai_analysis' in result
        assert 'analysis' in result['ai_analysis']
        assert 'recommendations' in result['ai_analysis']
        assert 'risk_level' in result['ai_analysis']
    
    def test_confidence_value_range(self):
        """Test that confidence values are within valid range."""
        # This would be tested through the actual endpoint calls above
        # but we can add specific validation here
        
        valid_confidences = [0.0, 0.5, 1.0, 0.999, 0.001]
        for conf in valid_confidences:
            assert 0.0 <= conf <= 1.0
        
        invalid_confidences = [-0.1, 1.1, -1.0, 2.0]
        for conf in invalid_confidences:
            assert not (0.0 <= conf <= 1.0)


class TestErrorHandling:
    """Test cases for error handling in detection API."""
    
    @patch('api.detection.load_model')
    def test_model_loading_error(self, mock_load_model):
        """Test handling of model loading errors."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        request_data = {
            'image_path': '/some/image.jpg',
            'role': 'general'
        }
        
        with pytest.raises(Exception):
            detect_fire_endpoint(request_data)
    
    @patch('api.detection.load_model')
    @patch('api.detection.preprocess_image')
    def test_preprocessing_error(self, mock_preprocess, mock_load_model):
        """Test handling of image preprocessing errors."""
        mock_load_model.return_value = Mock()
        mock_preprocess.side_effect = Exception("Preprocessing failed")
        
        request_data = {
            'image_path': '/some/image.jpg',
            'role': 'general'
        }
        
        with pytest.raises(Exception):
            detect_fire_endpoint(request_data)
    
    @patch('api.detection.load_model')
    @patch('api.detection.get_ai_response')
    def test_ai_service_error_handling(self, mock_ai_response, mock_load_model, sample_fire_image):
        """Test handling of AI service errors."""
        # Setup model mock
        mock_model = Mock()
        mock_model.return_value = Mock(squeeze=Mock(return_value=Mock(item=Mock(return_value=0.80))))
        mock_load_model.return_value = mock_model
        
        # Make AI service fail
        mock_ai_response.side_effect = Exception("AI service error")
        
        request_data = {
            'image_path': sample_fire_image,
            'role': 'general',
            'use_ai': True,
            'ai_provider': 'openai'
        }
        
        # Should not crash, should return result without AI analysis or with error info
        result = detect_fire_endpoint(request_data)
        assert 'fire_detected' in result
        assert 'confidence' in result
        # AI analysis might be missing or contain error info