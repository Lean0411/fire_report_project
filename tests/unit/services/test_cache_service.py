"""
測試 CacheService
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
import pickle
import json
from datetime import datetime, timedelta
from services.cache_service import CacheService


class TestCacheService:
    """CacheService 測試類"""
    
    @pytest.fixture
    def cache_service(self):
        """創建 CacheService 實例（使用記憶體快取）"""
        return CacheService()
    
    @pytest.fixture
    def redis_cache_service(self):
        """創建帶 Redis 的 CacheService 實例"""
        with patch('services.cache_service.redis'):
            service = CacheService('redis://localhost:6379')
            service.redis_client = MagicMock()
            return service
    
    def test_init_without_redis(self, cache_service):
        """測試沒有 Redis 的初始化"""
        assert cache_service.redis_client is None
        assert cache_service.memory_cache == {}
        assert cache_service.cache_stats == {'hits': 0, 'misses': 0, 'sets': 0}
    
    @patch('services.cache_service.REDIS_AVAILABLE', True)
    @patch('services.cache_service.redis')
    def test_init_with_redis_success(self, mock_redis):
        """測試成功初始化 Redis"""
        mock_client = MagicMock()
        mock_redis.from_url.return_value = mock_client
        mock_client.ping.return_value = True
        
        service = CacheService('redis://localhost:6379')
        
        assert service.redis_client == mock_client
        mock_redis.from_url.assert_called_once()
        mock_client.ping.assert_called_once()
    
    def test_get_memory_cache_hit(self, cache_service):
        """測試記憶體快取命中"""
        # 設置快取
        cache_service.memory_cache['test_key'] = {
            'value': 'test_value',
            'expires_at': datetime.utcnow() + timedelta(seconds=60)
        }
        
        result = cache_service.get('test_key')
        
        assert result == 'test_value'
        assert cache_service.cache_stats['hits'] == 1
        assert cache_service.cache_stats['misses'] == 0
    
    def test_get_memory_cache_miss(self, cache_service):
        """測試記憶體快取未命中"""
        result = cache_service.get('non_existent_key')
        
        assert result is None
        assert cache_service.cache_stats['hits'] == 0
        assert cache_service.cache_stats['misses'] == 1
    
    def test_get_memory_cache_expired(self, cache_service):
        """測試記憶體快取過期"""
        # 設置已過期的快取
        cache_service.memory_cache['test_key'] = {
            'value': 'test_value',
            'expires_at': datetime.utcnow() - timedelta(seconds=60)
        }
        
        result = cache_service.get('test_key')
        
        assert result is None
        assert 'test_key' not in cache_service.memory_cache
        assert cache_service.cache_stats['misses'] == 1
    
    def test_get_redis_cache_hit(self, redis_cache_service):
        """測試 Redis 快取命中"""
        redis_cache_service.redis_client.get.return_value = pickle.dumps('redis_value')
        
        result = redis_cache_service.get('test_key')
        
        assert result == 'redis_value'
        assert redis_cache_service.cache_stats['hits'] == 1
    
    def test_set_memory_cache(self, cache_service):
        """測試設置記憶體快取"""
        cache_service.set('test_key', 'test_value', ttl=3600)
        
        assert 'test_key' in cache_service.memory_cache
        assert cache_service.memory_cache['test_key']['value'] == 'test_value'
        assert cache_service.cache_stats['sets'] == 1
    
    def test_set_redis_cache(self, redis_cache_service):
        """測試設置 Redis 快取"""
        redis_cache_service.set('test_key', 'test_value', ttl=3600)
        
        redis_cache_service.redis_client.setex.assert_called_once()
        assert redis_cache_service.cache_stats['sets'] == 1
    
    def test_delete_memory_cache(self, cache_service):
        """測試刪除記憶體快取"""
        cache_service.memory_cache['test_key'] = {'value': 'test_value'}
        
        result = cache_service.delete('test_key')
        
        assert result is True
        assert 'test_key' not in cache_service.memory_cache
    
    def test_delete_redis_cache(self, redis_cache_service):
        """測試刪除 Redis 快取"""
        redis_cache_service.redis_client.delete.return_value = 1
        
        result = redis_cache_service.delete('test_key')
        
        assert result is True
        redis_cache_service.redis_client.delete.assert_called_once_with('test_key')
    
    def test_clear_memory_cache(self, cache_service):
        """測試清空記憶體快取"""
        cache_service.memory_cache = {'key1': {}, 'key2': {}}
        
        cache_service.clear()
        
        assert cache_service.memory_cache == {}
    
    def test_clear_redis_cache(self, redis_cache_service):
        """測試清空 Redis 快取"""
        redis_cache_service.clear()
        
        redis_cache_service.redis_client.flushdb.assert_called_once()
    
    def test_cleanup_expired_memory(self, cache_service):
        """測試清理過期的記憶體快取"""
        # 添加過期和未過期的項目
        now = datetime.utcnow()
        cache_service.memory_cache = {
            'expired1': {'expires_at': now - timedelta(seconds=60)},
            'expired2': {'expires_at': now - timedelta(seconds=30)},
            'valid': {'expires_at': now + timedelta(seconds=60)}
        }
        
        removed = cache_service.cleanup_expired(max_age=0)
        
        assert removed == 2
        assert 'valid' in cache_service.memory_cache
        assert 'expired1' not in cache_service.memory_cache
        assert 'expired2' not in cache_service.memory_cache
    
    def test_get_stats(self, cache_service):
        """測試獲取統計資訊"""
        cache_service.cache_stats = {'hits': 10, 'misses': 5, 'sets': 15}
        
        stats = cache_service.get_stats()
        
        assert stats['hits'] == 10
        assert stats['misses'] == 5
        assert stats['sets'] == 15
        assert stats['hit_rate'] == 66.67
        assert stats['total_requests'] == 15
    
    def test_cache_key_generation(self, cache_service):
        """測試快取鍵生成"""
        key = cache_service._generate_cache_key('prefix', 'arg1', 'arg2', kwarg1='value1')
        
        assert key.startswith('prefix:')
        assert len(key) > len('prefix:')
    
    def test_cache_model_prediction(self, cache_service):
        """測試模型預測快取"""
        prediction = {'is_fire': True, 'confidence': 0.95}
        
        cache_service.cache_model_prediction('image_path.jpg', prediction)
        
        cached = cache_service.get_model_prediction('image_path.jpg')
        assert cached == prediction
    
    def test_cache_sop_response(self, cache_service):
        """測試 SOP 回應快取"""
        sop_data = {'title': '火災應急程序', 'content': '撤離步驟...'}
        
        cache_service.cache_sop_response('general', True, sop_data)
        
        cached = cache_service.get_sop_response('general', True)
        assert cached == sop_data