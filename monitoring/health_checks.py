import psutil
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from flask import current_app
import threading
import traceback

from data.database import db
from data.models.system_metrics import SystemMetrics
from services.cache_service import cache_service
from services.ai_service import AIService

class HealthChecker:
    """System health monitoring and checks"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self.check_interval = 60  # seconds
        self.monitoring_thread = None
        self.monitoring_active = False
    
    def register_check(self, name: str, check_function, critical: bool = False):
        """Register a health check"""
        self.checks[name] = {
            'function': check_function,
            'critical': critical,
            'last_result': None,
            'last_error': None
        }
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.checks:
            return {'status': 'error', 'message': f'Check {name} not found'}
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            result = check['function']()
            duration = (time.time() - start_time) * 1000  # Convert to ms
            
            check_result = {
                'status': 'healthy' if result.get('healthy', True) else 'unhealthy',
                'message': result.get('message', ''),
                'data': result.get('data', {}),
                'duration_ms': round(duration, 2),
                'timestamp': datetime.utcnow().isoformat(),
                'critical': check['critical']
            }
            
            check['last_result'] = check_result
            check['last_error'] = None
            
            return check_result
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            error_result = {
                'status': 'error',
                'message': f'Check failed: {str(e)}',
                'data': {'error_type': type(e).__name__, 'traceback': traceback.format_exc()},
                'duration_ms': round(duration, 2),
                'timestamp': datetime.utcnow().isoformat(),
                'critical': check['critical']
            }
            
            check['last_result'] = error_result
            check['last_error'] = str(e)
            
            return error_result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_status = 'healthy'
        
        for name in self.checks:
            result = self.run_check(name)
            results[name] = result
            
            # Determine overall status
            if result['status'] == 'error' or result['status'] == 'unhealthy':
                if self.checks[name]['critical']:
                    overall_status = 'critical'
                elif overall_status == 'healthy':
                    overall_status = 'degraded'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background health monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                results = self.run_all_checks()
                
                # Record system metrics
                SystemMetrics.record_memory_usage(
                    memory_mb=psutil.virtual_memory().used / 1024 / 1024,
                    process_name='health_monitor'
                )
                
                # Log critical issues
                for check_name, result in results['checks'].items():
                    if result['status'] in ['error', 'unhealthy'] and self.checks[check_name]['critical']:
                        current_app.logger.error(
                            f"Critical health check failed: {check_name} - {result['message']}"
                        )
                
            except Exception as e:
                current_app.logger.error(f"Health monitoring error: {str(e)}")
            
            time.sleep(self.check_interval)

# Initialize global health checker
health_checker = HealthChecker()

# Database Health Check
def check_database():
    """Check database connectivity and performance"""
    try:
        start_time = time.time()
        
        # Test basic connectivity
        db.session.execute('SELECT 1')
        db.session.commit()
        
        query_time = (time.time() - start_time) * 1000
        
        # Check for recent activity
        recent_detections = db.session.execute(
            "SELECT COUNT(*) FROM detection_history WHERE created_at > datetime('now', '-1 hour')"
        ).scalar()
        
        return {
            'healthy': True,
            'message': 'Database is responsive',
            'data': {
                'query_time_ms': round(query_time, 2),
                'recent_detections': recent_detections
            }
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'message': f'Database connection failed: {str(e)}',
            'data': {'error': str(e)}
        }

def check_cache():
    """Check cache service health"""
    try:
        test_key = 'health_check_test'
        test_value = {'timestamp': datetime.utcnow().isoformat()}
        
        # Test set operation
        set_success = cache_service.set(test_key, test_value, ttl=60, namespace='health')
        
        # Test get operation
        retrieved_value = cache_service.get(test_key, namespace='health')
        
        # Test delete operation
        delete_success = cache_service.delete(test_key, namespace='health')
        
        # Get cache stats
        stats = cache_service.get_stats()
        
        if set_success and retrieved_value and delete_success:
            return {
                'healthy': True,
                'message': 'Cache service is working',
                'data': {
                    'cache_type': stats['cache_type'],
                    'hit_rate': stats['hit_rate'],
                    'total_operations': stats['hits'] + stats['misses'] + stats['sets']
                }
            }
        else:
            return {
                'healthy': False,
                'message': 'Cache operations failed',
                'data': {
                    'set_success': set_success,
                    'get_success': retrieved_value is not None,
                    'delete_success': delete_success
                }
            }
            
    except Exception as e:
        return {
            'healthy': False,
            'message': f'Cache check failed: {str(e)}',
            'data': {'error': str(e)}
        }

def check_ai_service():
    """Check AI service availability"""
    try:
        ai_service = AIService()
        
        # Test model loading (without actual inference)
        model_status = ai_service.get_model_status()
        
        # Check if AI providers are configured
        providers_status = {
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'ollama_configured': ai_service.check_ollama_availability()
        }
        
        any_provider_available = any(providers_status.values())
        
        return {
            'healthy': model_status.get('loaded', False) and any_provider_available,
            'message': 'AI services are ready' if any_provider_available else 'No AI providers configured',
            'data': {
                'model_status': model_status,
                'providers': providers_status
            }
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'message': f'AI service check failed: {str(e)}',
            'data': {'error': str(e)}
        }

def check_system_resources():
    """Check system resource usage"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / 1024 / 1024 / 1024
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = disk.free / 1024 / 1024 / 1024
        
        # Determine health status
        healthy = True
        warnings = []
        
        if cpu_percent > 90:
            healthy = False
            warnings.append('High CPU usage')
        elif cpu_percent > 70:
            warnings.append('Elevated CPU usage')
        
        if memory_percent > 90:
            healthy = False
            warnings.append('High memory usage')
        elif memory_percent > 70:
            warnings.append('Elevated memory usage')
        
        if disk_percent > 95:
            healthy = False
            warnings.append('Disk space critical')
        elif disk_percent > 80:
            warnings.append('Low disk space')
        
        message = 'System resources are normal'
        if warnings:
            message = f"Resource warnings: {', '.join(warnings)}"
        if not healthy:
            message = f"Resource issues detected: {', '.join(warnings)}"
        
        return {
            'healthy': healthy,
            'message': message,
            'data': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': round(memory_available_gb, 2),
                'disk_percent': disk_percent,
                'disk_free_gb': round(disk_free_gb, 2),
                'warnings': warnings
            }
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'message': f'System resource check failed: {str(e)}',
            'data': {'error': str(e)}
        }

def check_file_system():
    """Check file system and upload directory"""
    try:
        upload_dir = 'uploads'
        static_dir = 'static'
        
        issues = []
        
        # Check upload directory
        if not os.path.exists(upload_dir):
            issues.append(f'Upload directory {upload_dir} does not exist')
        elif not os.access(upload_dir, os.W_OK):
            issues.append(f'Upload directory {upload_dir} is not writable')
        
        # Check static directory
        if not os.path.exists(static_dir):
            issues.append(f'Static directory {static_dir} does not exist')
        
        # Check disk space in upload directory
        if os.path.exists(upload_dir):
            upload_disk = psutil.disk_usage(upload_dir)
            if upload_disk.free < 1024 * 1024 * 1024:  # Less than 1GB
                issues.append('Low disk space in upload directory')
        
        healthy = len(issues) == 0
        
        return {
            'healthy': healthy,
            'message': 'File system is accessible' if healthy else f"File system issues: {', '.join(issues)}",
            'data': {
                'upload_dir_exists': os.path.exists(upload_dir),
                'upload_dir_writable': os.access(upload_dir, os.W_OK) if os.path.exists(upload_dir) else False,
                'static_dir_exists': os.path.exists(static_dir),
                'issues': issues
            }
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'message': f'File system check failed: {str(e)}',
            'data': {'error': str(e)}
        }

# Register all health checks
health_checker.register_check('database', check_database, critical=True)
health_checker.register_check('cache', check_cache, critical=False)
health_checker.register_check('ai_service', check_ai_service, critical=True)
health_checker.register_check('system_resources', check_system_resources, critical=False)
health_checker.register_check('file_system', check_file_system, critical=True)