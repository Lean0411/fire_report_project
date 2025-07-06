#!/usr/bin/env python3
"""
Test script to verify constants configuration
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.constants import *

def test_constants():
    """Test that all constants are properly defined"""
    print("Testing constants configuration...")
    print("-" * 50)
    
    # Image processing constants
    print("Image Processing Constants:")
    print(f"  IMAGE_QUALITY_HIGH: {IMAGE_QUALITY_HIGH}")
    print(f"  IMAGE_QUALITY_MEDIUM: {IMAGE_QUALITY_MEDIUM}")
    print(f"  IMAGE_MAX_SIZE: {IMAGE_MAX_SIZE}")
    print(f"  AI_IMAGE_MAX_DIMENSION: {AI_IMAGE_MAX_DIMENSION}")
    print()
    
    # Model constants
    print("Model Constants:")
    print(f"  CNN_CHANNELS: {CNN_CHANNELS_INPUT} -> {CNN_CHANNELS_L1} -> {CNN_CHANNELS_L2} -> {CNN_CHANNELS_L3} -> {CNN_CHANNELS_L4}")
    print(f"  CNN_HIDDEN_SIZE: {CNN_HIDDEN_SIZE}")
    print(f"  CNN_DROPOUT_RATE: {CNN_DROPOUT_RATE}")
    print()
    
    # Cache constants
    print("Cache Constants:")
    print(f"  CACHE_TTL_MODEL_PREDICTIONS: {CACHE_TTL_MODEL_PREDICTIONS}s ({CACHE_TTL_MODEL_PREDICTIONS/3600}h)")
    print(f"  CACHE_TTL_SOP_RESPONSES: {CACHE_TTL_SOP_RESPONSES}s ({CACHE_TTL_SOP_RESPONSES/3600}h)")
    print()
    
    # Rate limiting constants
    print("Rate Limiting Constants:")
    print(f"  RATE_LIMIT_DEFAULT: {RATE_LIMIT_DEFAULT} req/min")
    print(f"  RATE_LIMIT_API: {RATE_LIMIT_API} req/min")
    print(f"  RATE_LIMIT_STRICT: {RATE_LIMIT_STRICT} req/min")
    print(f"  RATE_LIMIT_PUBLIC: {RATE_LIMIT_PUBLIC} req/min")
    print()
    
    # Authentication constants
    print("Authentication Constants:")
    print(f"  JWT_ACCESS_TOKEN_EXPIRES: {JWT_ACCESS_TOKEN_EXPIRES}")
    print(f"  JWT_REFRESH_TOKEN_EXPIRES: {JWT_REFRESH_TOKEN_EXPIRES}")
    print()
    
    # Server constants
    print("Server Constants:")
    print(f"  DEFAULT_PORT: {DEFAULT_PORT}")
    print(f"  DEFAULT_WORKERS: {DEFAULT_WORKERS}")
    print(f"  WORKER_TIMEOUT: {WORKER_TIMEOUT}s")
    print()
    
    # Application constants
    print("Application Constants:")
    print(f"  MAX_FILE_SIZE: {MAX_FILE_SIZE} bytes ({MAX_FILE_SIZE/BYTES_TO_MB}MB)")
    print(f"  API_VERSION: {API_VERSION}")
    print(f"  CORS_ALLOWED_PORTS: {CORS_ALLOWED_PORTS}")
    
    print("-" * 50)
    print("All constants loaded successfully!")

if __name__ == "__main__":
    test_constants()