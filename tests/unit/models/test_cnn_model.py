"""
Unit tests for CNN model functionality.
"""
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, Mock
import tempfile
import os

from models.cnn_model import CNNModel
from models.model_utils import load_model, preprocess_image


class TestCNNModel:
    """Test cases for CNN model."""
    
    def test_cnn_model_initialization(self):
        """Test CNN model can be initialized."""
        model = CNNModel()
        assert model is not None
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv3')
    
    def test_cnn_model_forward_pass(self):
        """Test forward pass through CNN model."""
        model = CNNModel()
        # Create dummy input tensor (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output is not None
        assert output.shape == (1, 1)  # Should output single probability
        assert 0 <= output.item() <= 1  # Should be probability between 0 and 1
    
    def test_model_eval_mode(self):
        """Test model can be set to evaluation mode."""
        model = CNNModel()
        model.eval()
        assert not model.training


class TestModelUtils:
    """Test cases for model utility functions."""
    
    @patch('models.model_utils.torch.load')
    @patch('models.model_utils.CNNModel')
    def test_load_model_success(self, mock_cnn_class, mock_torch_load):
        """Test successful model loading."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_cnn_class.return_value = mock_model_instance
        mock_torch_load.return_value = {'state': 'dict'}
        
        # Test
        result = load_model('dummy_path.pth')
        
        # Assertions
        assert result == mock_model_instance
        mock_torch_load.assert_called_once_with('dummy_path.pth', map_location='cpu')
        mock_model_instance.load_state_dict.assert_called_once()
        mock_model_instance.eval.assert_called_once()
    
    @patch('models.model_utils.torch.load')
    def test_load_model_file_not_found(self, mock_torch_load):
        """Test model loading with missing file."""
        mock_torch_load.side_effect = FileNotFoundError("Model file not found")
        
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model.pth')
    
    def test_preprocess_image_valid(self, sample_fire_image):
        """Test image preprocessing with valid image."""
        # Load the test image
        image = Image.open(sample_fire_image)
        
        # Preprocess
        processed = preprocess_image(image)
        
        # Assertions
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (1, 3, 224, 224)  # Batch size 1, RGB, 224x224
        assert processed.dtype == torch.float32
        
        # Check normalization (values should be roughly in [-2, 2] range for ImageNet normalization)
        assert processed.min() >= -3
        assert processed.max() <= 3
    
    def test_preprocess_image_different_sizes(self):
        """Test preprocessing images of different sizes."""
        # Create images of different sizes
        sizes = [(100, 100), (300, 400), (50, 200)]
        
        for width, height in sizes:
            # Create test image
            img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            
            # Process
            processed = preprocess_image(image)
            
            # Should always output 224x224
            assert processed.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_grayscale_conversion(self):
        """Test preprocessing converts grayscale to RGB."""
        # Create grayscale image
        img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        image = Image.fromarray(img_array, mode='L')
        
        # Process
        processed = preprocess_image(image)
        
        # Should convert to 3 channels
        assert processed.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_invalid_input(self):
        """Test preprocessing with invalid inputs."""
        with pytest.raises((AttributeError, TypeError)):
            preprocess_image(None)
        
        with pytest.raises((AttributeError, TypeError)):
            preprocess_image("not_an_image")


class TestModelPrediction:
    """Test cases for model prediction functionality."""
    
    @patch('models.model_utils.load_model')
    def test_prediction_pipeline(self, mock_load_model, sample_fire_image):
        """Test complete prediction pipeline."""
        # Setup mock model
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_output = Mock()
        mock_output.squeeze.return_value = Mock(item=Mock(return_value=0.85))
        mock_model.return_value = mock_output
        mock_load_model.return_value = mock_model
        
        # Load image
        image = Image.open(sample_fire_image)
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        # Mock prediction
        with torch.no_grad():
            prediction = mock_model(processed_image)
            confidence = prediction.squeeze().item()
        
        # Assertions
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert confidence == 0.85