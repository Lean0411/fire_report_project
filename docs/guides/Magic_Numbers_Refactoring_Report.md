# Magic Numbers Refactoring Report

## Overview
This report documents the refactoring effort to eliminate magic numbers (hardcoded numeric values) from the FireGuard AI codebase and replace them with properly named constants in a centralized configuration module.

## Summary of Changes

### 1. Created Central Constants Configuration
- **File**: `/config/constants.py`
- **Purpose**: Centralize all numeric constants used throughout the application
- **Benefits**: 
  - Single source of truth for configuration values
  - Easy to modify and maintain
  - Clear naming improves code readability
  - Support for environment variable overrides

### 2. Refactored Files

#### Image Processing (`services/image_service.py`)
- **Replaced Magic Numbers**:
  - `95` → `IMAGE_QUALITY_HIGH` (JPEG quality for annotated/resized images)
  - `85` → `IMAGE_QUALITY_MEDIUM` (JPEG quality for web format)
  - `(1024, 1024)` → `IMAGE_MAX_SIZE` (default max image dimensions)
  - `20` → `FONT_SIZE_MIN` (minimum font size)
  - `40` → `FONT_SIZE_DIVISOR` (font size calculation divisor)
  - `10` → `UI_PADDING` (UI element padding)
  - RGB values → Named color constants (`COLOR_FIRE_RED`, `COLOR_SAFE_GREEN`, `COLOR_WHITE`)
  - `128` → `ALPHA_SEMI_TRANSPARENT` (alpha channel value)
  - `24` → `IMAGE_CLEANUP_MAX_AGE_HOURS` (cleanup age threshold)
  - `100` → `IMAGE_CLEANUP_MAX_FILES` (max files to keep)

#### AI Service (`services/ai_service.py`)
- **Replaced Magic Numbers**:
  - `1000` → `OPENAI_MAX_TOKENS` (OpenAI API max tokens)
  - `0.7` → `OPENAI_TEMPERATURE` (OpenAI temperature setting)
  - `500` → `OLLAMA_MAX_TOKENS` (Ollama API max tokens)
  - `0.7` → `OLLAMA_TEMPERATURE` (Ollama temperature setting)
  - `30` → `AI_REQUEST_TIMEOUT` (HTTP request timeout)
  - `2048` → `AI_IMAGE_MAX_DIMENSION` (max image dimension for OpenAI)
  - `85` → `IMAGE_QUALITY_MEDIUM` (JPEG quality for AI processing)

#### CNN Model (`models/cnn_model.py`)
- **Replaced Magic Numbers**:
  - Channel dimensions: `3`, `64`, `128`, `256`, `512` → Named channel constants
  - `3` → `CNN_KERNEL_SIZE` (convolution kernel size)
  - `2` → `CNN_POOL_SIZE` (pooling kernel size)
  - `1024` → `CNN_HIDDEN_SIZE` (hidden layer size)
  - `0.5` → `CNN_DROPOUT_RATE` (dropout probability)
  - `14` → `CNN_FEATURE_MAP_SIZE` (feature map dimensions)
  - `(224, 224)` → `IMAGE_INPUT_SIZE` (model input dimensions)

#### Model Utils (`models/model_utils.py`)
- **Replaced Magic Numbers**:
  - `(224, 224)` → `IMAGE_INPUT_SIZE` (image preprocessing size)
  - `[0.485, 0.456, 0.406]` → `IMAGE_NORMALIZE_MEAN` (normalization means)
  - `[0.229, 0.224, 0.225]` → `IMAGE_NORMALIZE_STD` (normalization stds)

#### Cache Service (`services/cache_service.py`)
- **Replaced Magic Numbers**:
  - `3600` → `CACHE_TTL_MODEL_PREDICTIONS` (model cache TTL)
  - `7200` → `CACHE_TTL_SOP_RESPONSES` (SOP cache TTL)
  - `100` → `PERCENT_MULTIPLIER` (percentage calculation)

#### Rate Limiting (`api/middleware/rate_limit.py`)
- **Replaced Magic Numbers**:
  - `60` → `RATE_LIMIT_DEFAULT` (default rate limit)
  - `100` → `RATE_LIMIT_API` (API endpoint rate limit)
  - `10` → `RATE_LIMIT_STRICT` (strict rate limit)
  - `30` → `RATE_LIMIT_PUBLIC` (public endpoint rate limit)
  - `60` → `RATE_LIMIT_WINDOW` (rate limit time window)
  - `3600` → `CACHE_CLEANUP_MAX_AGE` (cleanup threshold)

#### Authentication Service (`services/auth/auth_service.py`)
- **Replaced Magic Numbers**:
  - `timedelta(hours=24)` → `JWT_ACCESS_TOKEN_EXPIRES`
  - `timedelta(days=30)` → `JWT_REFRESH_TOKEN_EXPIRES`
  - `7` → `AUTH_BEARER_PREFIX_LENGTH` (length of "Bearer " prefix)

#### Gunicorn Configuration (`gunicorn.conf.py`)
- **Replaced Magic Numbers**:
  - `'4'` → `DEFAULT_WORKERS` (default worker count)
  - `1000` → `WORKER_CONNECTIONS`
  - `30` → `WORKER_TIMEOUT`
  - `2` → `WORKER_KEEPALIVE`
  - `1000` → `MAX_REQUESTS`
  - `50` → `MAX_REQUESTS_JITTER`
  - `4094` → `LIMIT_REQUEST_LINE`
  - `100` → `LIMIT_REQUEST_FIELDS`
  - `8190` → `LIMIT_REQUEST_FIELD_SIZE`

#### Main Application (`app.py`)
- **Replaced Magic Numbers**:
  - `5002` → `DEFAULT_PORT` (default application port)
  - `[3000, 5000]` → `CORS_ALLOWED_PORTS` (allowed CORS ports)
  - `'2.0.0'` → `API_VERSION` (API version string)
  - HTTP status codes → Named constants (`HTTP_NOT_FOUND`, `HTTP_INTERNAL_ERROR`, `HTTP_RATE_LIMITED`)

#### Configuration Settings (`config/settings.py`)
- **Replaced Magic Numbers**:
  - `5 * 1024 * 1024` → `MAX_FILE_SIZE` (5MB file size limit)
  - `10` → `MIN_API_KEY_LENGTH` (minimum API key length)

## Benefits Achieved

1. **Improved Maintainability**: All configuration values are now in one place, making it easy to update them without searching through the codebase.

2. **Better Readability**: Named constants clearly indicate their purpose, making the code self-documenting.

3. **Reduced Duplication**: Eliminated duplicate definitions (e.g., `MAX_FILE_SIZE` was defined in multiple places).

4. **Environment Variable Support**: The constants module includes a `get_constant()` function that allows environment variables to override default values.

5. **Type Safety**: Constants are properly typed and grouped by category.

6. **Easy Testing**: Created a test script to verify all constants are properly loaded and accessible.

## Usage Examples

```python
# Before
img.save(annotated_path, quality=95)

# After
from config.constants import IMAGE_QUALITY_HIGH
img.save(annotated_path, quality=IMAGE_QUALITY_HIGH)
```

```python
# Before
nn.Conv2d(3, 64, 3, padding=1)

# After
from config.constants import CNN_CHANNELS_INPUT, CNN_CHANNELS_L1, CNN_KERNEL_SIZE
nn.Conv2d(CNN_CHANNELS_INPUT, CNN_CHANNELS_L1, CNN_KERNEL_SIZE, padding=1)
```

## Environment Variable Overrides

Constants can be overridden using environment variables with the `APP_` prefix:

```bash
# Override default port
export APP_DEFAULT_PORT=8080

# Override image quality
export APP_IMAGE_QUALITY_HIGH=100

# Override rate limits
export APP_RATE_LIMIT_API=200
```

## Recommendations

1. **Document Constants**: Add comments explaining the purpose and valid ranges for each constant.

2. **Validation**: Consider adding validation for constants that have specific valid ranges.

3. **Configuration Profiles**: Create different configuration profiles for development, testing, and production environments.

4. **Dynamic Reloading**: Implement a mechanism to reload certain constants without restarting the application.

5. **Monitoring**: Add monitoring for configuration-related metrics (e.g., cache hit rates, rate limit violations).

## Conclusion

This refactoring effort has successfully eliminated magic numbers from the codebase, improving maintainability, readability, and flexibility. The centralized constants configuration provides a solid foundation for future development and deployment configurations.