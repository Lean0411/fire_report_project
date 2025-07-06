import json
import hashlib
import pickle
from typing import Any, Optional, Dict, Union, Callable
from datetime import datetime, timedelta
import logging
from config.constants import (
    CACHE_TTL_MODEL_PREDICTIONS, CACHE_TTL_SOP_RESPONSES,
    CACHE_CLEANUP_MAX_AGE, PERCENT_MULTIPLIER
)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class CacheService:
    """Centralized caching service with Redis fallback to in-memory"""
    
    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'sets': 0}
        
        # Try to initialize Redis
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                # Test connection
                self.redis_client.ping()
                self.logger.info("Redis cache initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}, falling back to memory cache")
                self.redis_client = None
        else:
            self.logger.info("Redis not available, using memory cache")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value).encode()
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first (for simple types)
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key with namespace"""
        return f"{namespace}:{key}"
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                data = self.redis_client.get(cache_key)
                if data is not None:
                    self.cache_stats['hits'] += 1
                    return self._deserialize(data)
            else:
                # Memory cache
                if cache_key in self.memory_cache:
                    cache_entry = self.memory_cache[cache_key]
                    if cache_entry['expires_at'] is None or cache_entry['expires_at'] > datetime.utcnow():
                        self.cache_stats['hits'] += 1
                        return cache_entry['value']
                    else:
                        # Expired, remove from cache
                        del self.memory_cache[cache_key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {cache_key}: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "default") -> bool:
        """Set value in cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                serialized_value = self._serialize(value)
                if ttl:
                    result = self.redis_client.setex(cache_key, ttl, serialized_value)
                else:
                    result = self.redis_client.set(cache_key, serialized_value)
                
                if result:
                    self.cache_stats['sets'] += 1
                    return True
            else:
                # Memory cache
                expires_at = None
                if ttl:
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                self.memory_cache[cache_key] = {
                    'value': value,
                    'expires_at': expires_at,
                    'created_at': datetime.utcnow()
                }
                self.cache_stats['sets'] += 1
                return True
                
        except Exception as e:
            self.logger.error(f"Cache set error for key {cache_key}: {e}")
            
        return False
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                result = self.redis_client.delete(cache_key)
                return result > 0
            else:
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    return True
                    
        except Exception as e:
            self.logger.error(f"Cache delete error for key {cache_key}: {e}")
            
        return False
    
    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in namespace"""
        try:
            if self.redis_client:
                pattern = f"{namespace}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                return True
            else:
                keys_to_delete = [key for key in self.memory_cache.keys() if key.startswith(f"{namespace}:")]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                return True
                
        except Exception as e:
            self.logger.error(f"Cache clear namespace error for {namespace}: {e}")
            
        return False
    
    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                return self.redis_client.exists(cache_key) > 0
            else:
                if cache_key in self.memory_cache:
                    cache_entry = self.memory_cache[cache_key]
                    if cache_entry['expires_at'] is None or cache_entry['expires_at'] > datetime.utcnow():
                        return True
                    else:
                        del self.memory_cache[cache_key]
                        
        except Exception as e:
            self.logger.error(f"Cache exists error for key {cache_key}: {e}")
            
        return False
    
    def increment(self, key: str, amount: int = 1, namespace: str = "default") -> Optional[int]:
        """Increment numeric value in cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_client:
                return self.redis_client.incrby(cache_key, amount)
            else:
                current_value = self.get(key, namespace) or 0
                new_value = current_value + amount
                if self.set(key, new_value, namespace=namespace):
                    return new_value
                    
        except Exception as e:
            self.logger.error(f"Cache increment error for key {cache_key}: {e}")
            
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache_stats.copy()
        stats['cache_type'] = 'redis' if self.redis_client else 'memory'
        
        if not self.redis_client:
            stats['memory_keys'] = len(self.memory_cache)
            # Clean expired keys and count valid ones
            now = datetime.utcnow()
            expired_keys = []
            valid_keys = 0
            
            for cache_key, cache_entry in self.memory_cache.items():
                if cache_entry['expires_at'] and cache_entry['expires_at'] <= now:
                    expired_keys.append(cache_key)
                else:
                    valid_keys += 1
            
            # Clean expired keys
            for key in expired_keys:
                del self.memory_cache[key]
            
            stats['valid_keys'] = valid_keys
            stats['expired_cleaned'] = len(expired_keys)
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = (stats['hits'] / total_requests * PERCENT_MULTIPLIER) if total_requests > 0 else 0
        
        return stats
    
    def cleanup_expired(self) -> int:
        """Cleanup expired entries (for memory cache)"""
        if self.redis_client:
            return 0  # Redis handles expiration automatically
        
        now = datetime.utcnow()
        expired_keys = []
        
        for cache_key, cache_entry in self.memory_cache.items():
            if cache_entry['expires_at'] and cache_entry['expires_at'] <= now:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        return len(expired_keys)

class ModelCacheService:
    """Specialized cache service for ML model predictions"""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
        self.namespace = "model_predictions"
        self.default_ttl = CACHE_TTL_MODEL_PREDICTIONS  # 1 hour
    
    def get_prediction_key(self, image_hash: str, model_version: str) -> str:
        """Generate cache key for model prediction"""
        return f"{model_version}:{image_hash}"
    
    def get_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image data"""
        return hashlib.md5(image_data).hexdigest()
    
    def cache_prediction(self, image_data: bytes, model_version: str, 
                        prediction: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache model prediction"""
        image_hash = self.get_image_hash(image_data)
        cache_key = self.get_prediction_key(image_hash, model_version)
        
        cache_data = {
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': model_version
        }
        
        return self.cache.set(
            key=cache_key,
            value=cache_data,
            ttl=ttl or self.default_ttl,
            namespace=self.namespace
        )
    
    def get_cached_prediction(self, image_data: bytes, model_version: str) -> Optional[Dict[str, Any]]:
        """Get cached model prediction"""
        image_hash = self.get_image_hash(image_data)
        cache_key = self.get_prediction_key(image_hash, model_version)
        
        return self.cache.get(key=cache_key, namespace=self.namespace)
    
    def invalidate_model_cache(self, model_version: str) -> bool:
        """Invalidate all cache entries for a specific model version"""
        # This is a simplified implementation
        # In production, you might want to use Redis patterns for efficient deletion
        return self.cache.clear_namespace(f"{self.namespace}:{model_version}")

class SopCacheService:
    """Specialized cache service for SOP (Standard Operating Procedures) data"""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
        self.namespace = "sop_data"
        self.default_ttl = CACHE_TTL_SOP_RESPONSES  # 2 hours
    
    def cache_sop_response(self, query: str, role: str, location: str, 
                          response: str, ttl: Optional[int] = None) -> bool:
        """Cache SOP response"""
        cache_key = self._generate_sop_key(query, role, location)
        
        cache_data = {
            'response': response,
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'role': role,
            'location': location
        }
        
        return self.cache.set(
            key=cache_key,
            value=cache_data,
            ttl=ttl or self.default_ttl,
            namespace=self.namespace
        )
    
    def get_cached_sop_response(self, query: str, role: str, location: str) -> Optional[str]:
        """Get cached SOP response"""
        cache_key = self._generate_sop_key(query, role, location)
        cached_data = self.cache.get(key=cache_key, namespace=self.namespace)
        
        if cached_data:
            return cached_data.get('response')
        
        return None
    
    def _generate_sop_key(self, query: str, role: str, location: str) -> str:
        """Generate cache key for SOP query"""
        key_data = f"{query}:{role}:{location}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def invalidate_sop_cache(self) -> bool:
        """Invalidate all SOP cache entries"""
        return self.cache.clear_namespace(self.namespace)

# Global cache service instances
cache_service = CacheService()
model_cache = ModelCacheService(cache_service)
sop_cache = SopCacheService(cache_service)