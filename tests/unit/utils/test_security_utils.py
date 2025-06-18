"""
Unit tests for security utility functions.
"""
import pytest
import tempfile
import os
from PIL import Image
import numpy as np

try:
    from utils.security_utils import (
        sanitize_string_input,
        validate_role,
        validate_ai_provider,
        validate_probability,
        validate_boolean,
        validate_filename,
        validate_api_key,
        validate_url
    )
except ImportError:
    pytest.skip("utils.security_utils module not available", allow_module_level=True)


@pytest.mark.unit
class TestStringInput:
    """Test cases for string input sanitization."""
    
    def test_sanitize_string_input_normal(self):
        """Test sanitization with normal strings."""
        assert sanitize_string_input("hello") == "hello"
        assert sanitize_string_input("test123") == "test123"
        assert sanitize_string_input("  spaced  ") == "spaced"
    
    def test_sanitize_string_input_dangerous_chars(self):
        """Test sanitization removes dangerous characters."""
        result = sanitize_string_input("<script>alert('xss')</script>")
        assert "<script>" not in result
        # After HTML escaping, dangerous chars should be encoded
        assert "&lt;" in result or "&gt;" in result  # HTML entities
    
    def test_sanitize_string_input_max_length(self):
        """Test max length enforcement."""
        long_string = "a" * 200
        result = sanitize_string_input(long_string, max_length=50)
        assert result == ""  # Should return default when too long
    
    def test_sanitize_string_input_non_string(self):
        """Test with non-string input."""
        assert sanitize_string_input(123) == ""
        assert sanitize_string_input(None) == ""
        assert sanitize_string_input([]) == ""


@pytest.mark.unit
class TestRoleValidation:
    """Test cases for role validation."""
    
    def test_validate_role_valid(self):
        """Test validation with valid roles."""
        # Note: This test depends on constants.py having USER_ROLES defined
        result = validate_role("general")
        assert result in ["general", "firefighter", "management"]
    
    def test_validate_role_invalid(self):
        """Test validation with invalid roles."""
        assert validate_role("invalid_role") == "general"
        assert validate_role("admin") == "general"
        assert validate_role("") == "general"
    
    def test_validate_role_non_string(self):
        """Test role validation with non-string input."""
        assert validate_role(123) == "general"
        assert validate_role(None) == "general"


@pytest.mark.unit  
class TestAIProviderValidation:
    """Test cases for AI provider validation."""
    
    def test_validate_ai_provider_valid(self):
        """Test validation with valid providers."""
        result = validate_ai_provider("openai")
        assert result in ["openai", "ollama"]
    
    def test_validate_ai_provider_invalid(self):
        """Test validation with invalid providers."""
        assert validate_ai_provider("invalid_provider") == "openai"
        assert validate_ai_provider("") == "openai"
    
    def test_validate_ai_provider_non_string(self):
        """Test provider validation with non-string input."""
        assert validate_ai_provider(123) == "openai"
        assert validate_ai_provider(None) == "openai"


@pytest.mark.unit
class TestProbabilityValidation:
    """Test cases for probability validation."""
    
    def test_validate_probability_valid_range(self):
        """Test validation with valid probability values."""
        assert validate_probability(0.0) == 0.0
        assert validate_probability(0.5) == 0.5
        assert validate_probability(1.0) == 1.0
        assert validate_probability(0.999) == 0.999
    
    def test_validate_probability_out_of_range(self):
        """Test validation with out-of-range values."""
        assert validate_probability(-0.1) == 0.0
        assert validate_probability(1.1) == 1.0
        assert validate_probability(-10) == 0.0
        assert validate_probability(5) == 1.0
    
    def test_validate_probability_non_numeric(self):
        """Test validation with non-numeric input."""
        assert validate_probability("not_a_number") == 0.0
        assert validate_probability(None) == 0.0
        assert validate_probability([]) == 0.0


@pytest.mark.unit
class TestBooleanValidation:
    """Test cases for boolean validation."""
    
    def test_validate_boolean_true_values(self):
        """Test validation with true values."""
        assert validate_boolean(True) == True
        assert validate_boolean("true") == True
        assert validate_boolean("True") == True
        assert validate_boolean("1") == True
        assert validate_boolean("yes") == True
        assert validate_boolean("on") == True
    
    def test_validate_boolean_false_values(self):
        """Test validation with false values."""
        assert validate_boolean(False) == False
        assert validate_boolean("false") == False
        assert validate_boolean("False") == False
        assert validate_boolean("0") == False
        assert validate_boolean("no") == False
        assert validate_boolean("off") == False
        assert validate_boolean("") == False
    
    def test_validate_boolean_non_boolean(self):
        """Test validation with non-boolean input."""
        assert validate_boolean(123) == False
        assert validate_boolean(None) == False
        assert validate_boolean([]) == False


@pytest.mark.unit
class TestFilenameValidation:
    """Test cases for filename validation."""
    
    def test_validate_filename_safe(self):
        """Test validation with safe filenames."""
        assert validate_filename("image.jpg") == True
        assert validate_filename("test_file.png") == True
        assert validate_filename("document123.pdf") == True
    
    def test_validate_filename_dangerous(self):
        """Test validation with dangerous filenames."""
        assert validate_filename("../etc/passwd") == False
        assert validate_filename("..\\windows\\system32") == False
        assert validate_filename("file/with/slashes.jpg") == False
        assert validate_filename("file\\with\\backslashes.jpg") == False
        assert validate_filename("file:with:colons.jpg") == False
    
    def test_validate_filename_too_long(self):
        """Test validation with overly long filenames."""
        long_filename = "a" * 300 + ".jpg"
        assert validate_filename(long_filename) == False
    
    def test_validate_filename_empty_or_none(self):
        """Test validation with empty or None filenames."""
        assert validate_filename("") == False
        assert validate_filename(None) == False


@pytest.mark.unit
class TestAPIKeyValidation:
    """Test cases for API key validation."""
    
    def test_validate_api_key_valid(self):
        """Test validation with valid API keys."""
        assert validate_api_key("sk-1234567890abcdef") == True
        assert validate_api_key("api_key_123456789") == True
        assert validate_api_key("1234567890abcdef1234567890") == True
    
    def test_validate_api_key_invalid(self):
        """Test validation with invalid API keys."""
        assert validate_api_key("short") == False
        assert validate_api_key("") == False
        assert validate_api_key(None) == False
        assert validate_api_key("key with spaces") == False
        assert validate_api_key("key@with#special$chars") == False
    
    def test_validate_api_key_non_string(self):
        """Test validation with non-string input."""
        assert validate_api_key(123456789) == False
        assert validate_api_key([]) == False


@pytest.mark.unit
class TestURLValidation:
    """Test cases for URL validation."""
    
    def test_validate_url_valid(self):
        """Test validation with valid URLs."""
        assert validate_url("https://example.com") == True
        assert validate_url("http://localhost:8080") == True
        assert validate_url("https://api.openai.com/v1/chat") == True
        assert validate_url("http://192.168.1.1:3000") == True
    
    def test_validate_url_invalid(self):
        """Test validation with invalid URLs."""
        assert validate_url("not_a_url") == False
        assert validate_url("ftp://example.com") == False
        assert validate_url("") == False
        assert validate_url(None) == False
        assert validate_url("javascript:alert('xss')") == False
    
    def test_validate_url_non_string(self):
        """Test validation with non-string input."""
        assert validate_url(123) == False
        assert validate_url([]) == False