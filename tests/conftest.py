"""
Simplified pytest configuration and shared fixtures for FireGuard AI tests.
"""
import os
import sys
import pytest
import tempfile
from PIL import Image
import numpy as np

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Set test environment
os.environ['TESTING'] = 'True'
os.environ['AI_PROVIDER'] = 'mock'

@pytest.fixture
def sample_fire_image(temp_file_tracker):
    """Create a sample fire image for testing."""
    # Create a red-orange image simulating fire
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = 255  # Red channel
    img_array[:, :, 1] = 165  # Green channel (orange)
    img_array[:, :, 2] = 0    # Blue channel
    
    img = Image.fromarray(img_array)
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name)
    temp_file.close()
    
    # Track for cleanup
    temp_file_tracker.append(temp_file.name)
    return temp_file.name

@pytest.fixture
def sample_no_fire_image(temp_file_tracker):
    """Create a sample non-fire image for testing."""
    # Create a blue-green image simulating normal scene
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = 0    # Red channel
    img_array[:, :, 1] = 128  # Green channel
    img_array[:, :, 2] = 255  # Blue channel
    
    img = Image.fromarray(img_array)
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name)
    temp_file.close()
    
    # Track for cleanup
    temp_file_tracker.append(temp_file.name)
    return temp_file.name

@pytest.fixture
def client():
    """Create Flask test client."""
    try:
        from app import app
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            yield client
    except ImportError:
        pytest.skip("Flask app not available")

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    temp_files = []
    yield temp_files
    # Cleanup
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except (OSError, TypeError):
            pass

@pytest.fixture
def temp_file_tracker():
    """Track temporary files for cleanup."""
    temp_files = []
    yield temp_files
    # Cleanup tracked files
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except (OSError, TypeError):
            pass