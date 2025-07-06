"""
Application Constants Configuration
Centralizes all numeric constants and magic numbers used throughout the application
"""
import os
from datetime import timedelta

# ============================================================================
# IMAGE PROCESSING CONSTANTS
# ============================================================================

# Image quality settings
IMAGE_QUALITY_HIGH = 95  # For annotated and resized images
IMAGE_QUALITY_MEDIUM = 85  # For web format and AI processing

# Image dimensions
IMAGE_MAX_SIZE = (1024, 1024)  # Maximum image size for resize
IMAGE_INPUT_SIZE = (224, 224)  # Model input image size
AI_IMAGE_MAX_DIMENSION = 2048  # Maximum dimension for OpenAI Vision API

# Font and UI settings
FONT_SIZE_MIN = 20  # Minimum font size for annotations
FONT_SIZE_DIVISOR = 40  # Divisor for dynamic font sizing (width / divisor)
UI_PADDING = 10  # Padding for UI elements

# Color definitions (RGB)
COLOR_FIRE_RED = (255, 0, 0)
COLOR_SAFE_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
ALPHA_SEMI_TRANSPARENT = 128

# File management
IMAGE_CLEANUP_MAX_AGE_HOURS = 24  # Hours before cleanup
IMAGE_CLEANUP_MAX_FILES = 100  # Maximum files to keep

# ============================================================================
# MODEL AND AI CONSTANTS
# ============================================================================

# Neural network architecture
CNN_CHANNELS_INPUT = 3
CNN_CHANNELS_L1 = 64
CNN_CHANNELS_L2 = 128
CNN_CHANNELS_L3 = 256
CNN_CHANNELS_L4 = 512
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
CNN_HIDDEN_SIZE = 1024
CNN_DROPOUT_RATE = 0.5
CNN_FEATURE_MAP_SIZE = 14  # Feature map size after convolutions

# Image normalization (ImageNet standards)
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]

# AI API settings
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.7
OLLAMA_MAX_TOKENS = 500
OLLAMA_TEMPERATURE = 0.7
AI_REQUEST_TIMEOUT = 30  # seconds

# ============================================================================
# CACHE AND TIMEOUT CONSTANTS
# ============================================================================

# Cache TTL (Time To Live) settings
CACHE_TTL_MODEL_PREDICTIONS = 3600  # 1 hour in seconds
CACHE_TTL_SOP_RESPONSES = 7200  # 2 hours in seconds
CACHE_CLEANUP_MAX_AGE = 3600  # 1 hour in seconds

# Rate limiting windows
RATE_LIMIT_WINDOW = 60  # 60 seconds window

# ============================================================================
# RATE LIMITING CONSTANTS
# ============================================================================

# Requests per minute limits
RATE_LIMIT_DEFAULT = 60  # Default rate limit
RATE_LIMIT_API = 100  # API endpoints
RATE_LIMIT_STRICT = 10  # Sensitive endpoints
RATE_LIMIT_PUBLIC = 30  # Public endpoints

# ============================================================================
# AUTHENTICATION CONSTANTS
# ============================================================================

# JWT token expiration
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)

# Authentication strings
AUTH_BEARER_PREFIX_LENGTH = 7  # Length of "Bearer " prefix
MIN_API_KEY_LENGTH = 10  # Minimum valid API key length

# ============================================================================
# SERVER CONFIGURATION CONSTANTS
# ============================================================================

# Gunicorn settings
DEFAULT_WORKERS = int(os.environ.get('WORKERS', '4'))
WORKER_CONNECTIONS = 1000
WORKER_TIMEOUT = 30
WORKER_KEEPALIVE = 2
MAX_REQUESTS = 1000
MAX_REQUESTS_JITTER = 50

# Request limits
LIMIT_REQUEST_LINE = 4094
LIMIT_REQUEST_FIELDS = 100
LIMIT_REQUEST_FIELD_SIZE = 8190

# ============================================================================
# APPLICATION CONSTANTS
# ============================================================================

# File size limits
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
BYTES_TO_MB = 1024 * 1024  # Conversion factor

# Ports
DEFAULT_PORT = 5002
CORS_ALLOWED_PORTS = [3000, 5000]

# API versioning
API_VERSION = '2.0.0'

# HTTP status codes (most common ones)
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_RATE_LIMITED = 429
HTTP_INTERNAL_ERROR = 500

# Percentage calculations
PERCENT_MULTIPLIER = 100

# ============================================================================
# ENVIRONMENT-BASED OVERRIDES
# ============================================================================

# Allow environment variables to override certain constants
def get_constant(name: str, default: any) -> any:
    """Get constant value with environment variable override support"""
    env_name = f"APP_{name}"
    env_value = os.environ.get(env_name)
    
    if env_value is not None:
        # Try to convert to appropriate type
        if isinstance(default, int):
            try:
                return int(env_value)
            except ValueError:
                pass
        elif isinstance(default, float):
            try:
                return float(env_value)
            except ValueError:
                pass
        elif isinstance(default, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        else:
            return env_value
    
    return default

# Example usage:
# IMAGE_QUALITY_HIGH = get_constant('IMAGE_QUALITY_HIGH', IMAGE_QUALITY_HIGH)