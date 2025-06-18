"""
Unit tests for AI service functionality.
"""
import pytest
from unittest.mock import patch, Mock
import json

from services.ai_service import get_ai_response, format_detection_prompt


class TestAIService:
    """Test cases for AI service."""
    
    @patch('services.ai_service.openai.ChatCompletion.create')
    def test_get_ai_response_openai_success(self, mock_openai):
        """Test successful OpenAI API response."""
        # Mock OpenAI response
        mock_response = {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'analysis': 'Fire detected with high confidence',
                        'recommendations': ['Evacuate immediately', 'Call 119'],
                        'risk_level': 'high'
                    })
                }
            }]
        }
        mock_openai.return_value = mock_response
        
        # Test
        result = get_ai_response(
            detection_result=True,
            confidence=0.95,
            role='firefighter',
            provider='openai'
        )
        
        # Assertions
        assert result is not None
        assert 'analysis' in result
        assert 'recommendations' in result
        assert 'risk_level' in result
        assert result['risk_level'] == 'high'
    
    @patch('services.ai_service.requests.post')
    def test_get_ai_response_ollama_success(self, mock_post):
        """Test successful Ollama API response."""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': json.dumps({
                'analysis': 'Fire detected with medium confidence',
                'recommendations': ['Monitor situation', 'Prepare evacuation'],
                'risk_level': 'medium'
            })
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Test
        result = get_ai_response(
            detection_result=True,
            confidence=0.75,
            role='general',
            provider='ollama'
        )
        
        # Assertions
        assert result is not None
        assert 'analysis' in result
        assert 'recommendations' in result
        assert result['risk_level'] == 'medium'
    
    def test_get_ai_response_invalid_provider(self):
        """Test AI response with invalid provider."""
        with pytest.raises(ValueError):
            get_ai_response(
                detection_result=True,
                confidence=0.8,
                role='general',
                provider='invalid_provider'
            )
    
    @patch('services.ai_service.openai.ChatCompletion.create')
    def test_get_ai_response_openai_error(self, mock_openai):
        """Test OpenAI API error handling."""
        mock_openai.side_effect = Exception("API Error")
        
        result = get_ai_response(
            detection_result=True,
            confidence=0.8,
            role='general',
            provider='openai'
        )
        
        # Should return fallback response
        assert result is not None
        assert 'error' in result or 'analysis' in result
    
    @patch('services.ai_service.requests.post')
    def test_get_ai_response_ollama_error(self, mock_post):
        """Test Ollama API error handling."""
        mock_post.side_effect = Exception("Connection Error")
        
        result = get_ai_response(
            detection_result=True,
            confidence=0.8,
            role='general',
            provider='ollama'
        )
        
        # Should return fallback response
        assert result is not None
        assert 'error' in result or 'analysis' in result


class TestPromptFormatting:
    """Test cases for prompt formatting."""
    
    def test_format_detection_prompt_fire_detected(self):
        """Test prompt formatting when fire is detected."""
        prompt = format_detection_prompt(
            detection_result=True,
            confidence=0.92,
            role='firefighter'
        )
        
        assert isinstance(prompt, str)
        assert 'fire detected' in prompt.lower()
        assert '0.92' in prompt or '92%' in prompt
        assert 'firefighter' in prompt.lower()
    
    def test_format_detection_prompt_no_fire(self):
        """Test prompt formatting when no fire is detected."""
        prompt = format_detection_prompt(
            detection_result=False,
            confidence=0.15,
            role='general'
        )
        
        assert isinstance(prompt, str)
        assert 'no fire' in prompt.lower() or 'safe' in prompt.lower()
        assert '0.15' in prompt or '15%' in prompt
    
    def test_format_detection_prompt_different_roles(self):
        """Test prompt formatting for different user roles."""
        roles = ['general', 'firefighter', 'management']
        
        for role in roles:
            prompt = format_detection_prompt(
                detection_result=True,
                confidence=0.8,
                role=role
            )
            
            assert isinstance(prompt, str)
            assert role in prompt.lower()
            assert len(prompt) > 50  # Should be a substantial prompt
    
    def test_format_detection_prompt_edge_cases(self):
        """Test prompt formatting with edge case values."""
        # Very high confidence
        prompt_high = format_detection_prompt(
            detection_result=True,
            confidence=0.99,
            role='general'
        )
        assert '0.99' in prompt_high or '99%' in prompt_high
        
        # Very low confidence
        prompt_low = format_detection_prompt(
            detection_result=False,
            confidence=0.01,
            role='general'
        )
        assert '0.01' in prompt_low or '1%' in prompt_low
    
    def test_format_detection_prompt_invalid_role(self):
        """Test prompt formatting with invalid role."""
        # Should handle gracefully or raise appropriate error
        try:
            prompt = format_detection_prompt(
                detection_result=True,
                confidence=0.8,
                role='invalid_role'
            )
            # If it doesn't raise an error, it should still return a valid prompt
            assert isinstance(prompt, str)
            assert len(prompt) > 0
        except ValueError:
            # This is also acceptable behavior
            pass


class TestAIResponseParsing:
    """Test cases for AI response parsing."""
    
    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response."""
        from services.ai_service import parse_ai_response
        
        valid_json = json.dumps({
            'analysis': 'Test analysis',
            'recommendations': ['Action 1', 'Action 2'],
            'risk_level': 'medium'
        })
        
        result = parse_ai_response(valid_json)
        
        assert result['analysis'] == 'Test analysis'
        assert len(result['recommendations']) == 2
        assert result['risk_level'] == 'medium'
    
    def test_parse_invalid_json_response(self):
        """Test parsing invalid JSON response."""
        from services.ai_service import parse_ai_response
        
        invalid_json = "This is not valid JSON"
        
        result = parse_ai_response(invalid_json)
        
        # Should return fallback structure
        assert 'analysis' in result
        assert 'recommendations' in result
        assert 'risk_level' in result
    
    def test_parse_partial_json_response(self):
        """Test parsing partial JSON response."""
        from services.ai_service import parse_ai_response
        
        partial_json = json.dumps({
            'analysis': 'Partial response'
            # Missing recommendations and risk_level
        })
        
        result = parse_ai_response(partial_json)
        
        assert result['analysis'] == 'Partial response'
        assert 'recommendations' in result  # Should have default
        assert 'risk_level' in result  # Should have default