from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from services.auth.auth_service import auth_service
from data.repositories.detection_repository import DetectionRepository
from data.models.system_metrics import SystemMetrics, ApiUsage
from data.repositories.user_repository import UserRepository
from monitoring.health_checks import health_checker
from services.cache_service import cache_service
import psutil

monitoring_bp = Blueprint('monitoring_v1', __name__, url_prefix='/api/v1/monitoring')

@monitoring_bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        # Run all health checks
        results = health_checker.run_all_checks()
        
        # Determine HTTP status code based on overall status
        status_codes = {
            'healthy': 200,
            'degraded': 200,  # Still operational
            'critical': 503   # Service unavailable
        }
        
        status_code = status_codes.get(results['overall_status'], 500)
        
        return jsonify(results), status_code
        
    except Exception as e:
        return jsonify({
            'overall_status': 'error',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500

@monitoring_bp.route('/health/<check_name>', methods=['GET'])
@auth_service.require_admin
def individual_health_check(check_name):
    """Run individual health check"""
    try:
        result = health_checker.run_check(check_name)
        
        if result['status'] == 'error':
            return jsonify(result), 404 if 'not found' in result['message'] else 500
        
        status_code = 200 if result['status'] == 'healthy' else 503
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@monitoring_bp.route('/metrics/system', methods=['GET'])
@auth_service.require_admin
def system_metrics():
    """Get current system metrics"""
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get cache statistics
        cache_stats = cache_service.get_stats()
        
        # Get recent error count
        hours = request.args.get('hours', 24, type=int)
        error_rate = SystemMetrics.get_error_rate(hours=hours)
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory': {
                    'total_gb': round(memory.total / 1024 / 1024 / 1024, 2),
                    'used_gb': round(memory.used / 1024 / 1024 / 1024, 2),
                    'available_gb': round(memory.available / 1024 / 1024 / 1024, 2),
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': round(disk.total / 1024 / 1024 / 1024, 2),
                    'used_gb': round(disk.used / 1024 / 1024 / 1024, 2),
                    'free_gb': round(disk.free / 1024 / 1024 / 1024, 2),
                    'percent': round(disk.used / disk.total * 100, 1)
                }
            },
            'cache': cache_stats,
            'error_rate_percent': round(error_rate, 2),
            'period_hours': hours
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/metrics/api', methods=['GET'])
@auth_service.require_admin
def api_metrics():
    """Get API usage metrics"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        # Get detection statistics
        detection_repo = DetectionRepository()
        detection_stats = detection_repo.get_detection_statistics(days=hours//24 or 1)
        
        # Get average response times
        avg_response_time = SystemMetrics.get_average_response_time(hours=hours)
        detection_response_time = SystemMetrics.get_average_response_time(
            endpoint='/api/v1/detection/analyze',
            hours=hours
        )
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'api_performance': {
                'average_response_time_ms': round(avg_response_time, 2),
                'detection_response_time_ms': round(detection_response_time, 2),
                'error_rate_percent': round(SystemMetrics.get_error_rate(hours=hours), 2)
            },
            'detection_stats': detection_stats
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/metrics/users', methods=['GET'])
@auth_service.require_admin
def user_metrics():
    """Get user activity metrics"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        user_repo = UserRepository()
        
        # Get user counts
        total_users = user_repo.count()
        active_users = len(user_repo.get_active_users())
        
        # Get users by role
        admin_count = len(user_repo.get_users_by_role('admin'))
        firefighter_count = len(user_repo.get_users_by_role('firefighter'))
        regular_user_count = len(user_repo.get_users_by_role('user'))
        
        # Get recent user activity
        start_time = datetime.utcnow() - timedelta(hours=hours)
        recent_logins = user_repo.model.query.filter(
            user_repo.model.last_login >= start_time
        ).count()
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'user_counts': {
                'total': total_users,
                'active': active_users,
                'inactive': total_users - active_users,
                'recent_logins': recent_logins
            },
            'role_distribution': {
                'admin': admin_count,
                'firefighter': firefighter_count,
                'user': regular_user_count
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/metrics/detections', methods=['GET'])
@auth_service.require_firefighter
def detection_metrics():
    """Get detailed detection metrics"""
    try:
        days = request.args.get('days', 7, type=int)
        
        detection_repo = DetectionRepository()
        
        # Get comprehensive statistics
        stats = detection_repo.get_detection_statistics(days=days)
        
        # Get daily counts
        daily_counts = detection_repo.get_daily_detection_counts(days=days)
        
        # Get recent fire detections
        recent_fires = detection_repo.get_recent_detections(hours=24, fire_only=True, limit=10)
        recent_fire_summaries = [detection.get_summary() for detection in recent_fires]
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'statistics': stats,
            'daily_trends': daily_counts,
            'recent_fire_detections': recent_fire_summaries
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/status', methods=['GET'])
def service_status():
    """Simple service status endpoint for load balancers"""
    try:
        # Quick health check
        from data.database import db
        db.session.execute('SELECT 1')
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'FireGuard AI',
            'version': '2.0.0'
        }), 200
        
    except Exception:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat()
        }), 503

@monitoring_bp.route('/alerts', methods=['GET'])
@auth_service.require_admin
def get_alerts():
    """Get system alerts and warnings"""
    try:
        # Run health checks to get current status
        health_results = health_checker.run_all_checks()
        
        alerts = []
        
        # Process health check results for alerts
        for check_name, result in health_results['checks'].items():
            if result['status'] in ['error', 'unhealthy']:
                severity = 'critical' if health_checker.checks[check_name]['critical'] else 'warning'
                alerts.append({
                    'type': 'health_check',
                    'severity': severity,
                    'source': check_name,
                    'message': result['message'],
                    'timestamp': result['timestamp'],
                    'data': result.get('data', {})
                })
        
        # Check for high error rates
        error_rate = SystemMetrics.get_error_rate(hours=1)
        if error_rate > 10:  # More than 10% error rate
            alerts.append({
                'type': 'performance',
                'severity': 'critical' if error_rate > 25 else 'warning',
                'source': 'error_rate',
                'message': f'High error rate: {error_rate:.1f}%',
                'timestamp': datetime.utcnow().isoformat(),
                'data': {'error_rate': error_rate}
            })
        
        # Check for recent fire detections (for awareness)
        detection_repo = DetectionRepository()
        recent_fires = detection_repo.get_recent_detections(hours=1, fire_only=True, limit=5)
        if recent_fires:
            alerts.append({
                'type': 'fire_detection',
                'severity': 'info',
                'source': 'detection_system',
                'message': f'{len(recent_fires)} fire(s) detected in the last hour',
                'timestamp': datetime.utcnow().isoformat(),
                'data': {'count': len(recent_fires)}
            })
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'alert_count': len(alerts),
            'alerts': alerts,
            'overall_status': health_results['overall_status']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500