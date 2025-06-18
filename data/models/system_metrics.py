from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import JSON
from ..database import db, BaseModel

class SystemMetrics(BaseModel):
    """Model for storing system performance and usage metrics"""
    __tablename__ = 'system_metrics'
    
    metric_type = db.Column(db.String(50), nullable=False)  # 'performance', 'usage', 'error', 'api'
    metric_name = db.Column(db.String(100), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    metric_unit = db.Column(db.String(20))  # 'ms', 'mb', 'count', 'percent'
    
    # Context information
    endpoint = db.Column(db.String(200))  # API endpoint if applicable
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    session_id = db.Column(db.String(100))
    
    # Additional metadata
    metadata = db.Column(JSON)  # Additional metric data
    tags = db.Column(JSON)  # Tags for categorization
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def record_api_response_time(cls, endpoint, response_time, user_id=None, session_id=None):
        """Record API response time metric"""
        metric = cls(
            metric_type='performance',
            metric_name='api_response_time',
            metric_value=response_time,
            metric_unit='ms',
            endpoint=endpoint,
            user_id=user_id,
            session_id=session_id
        )
        metric.save()
        return metric
    
    @classmethod
    def record_model_inference_time(cls, model_name, inference_time, user_id=None):
        """Record model inference time"""
        metric = cls(
            metric_type='performance',
            metric_name='model_inference_time',
            metric_value=inference_time,
            metric_unit='ms',
            user_id=user_id,
            metadata={'model_name': model_name}
        )
        metric.save()
        return metric
    
    @classmethod
    def record_memory_usage(cls, memory_mb, process_name=None):
        """Record memory usage"""
        metric = cls(
            metric_type='performance',
            metric_name='memory_usage',
            metric_value=memory_mb,
            metric_unit='mb',
            metadata={'process_name': process_name}
        )
        metric.save()
        return metric
    
    @classmethod
    def record_error(cls, error_type, endpoint=None, user_id=None, error_details=None):
        """Record error occurrence"""
        metric = cls(
            metric_type='error',
            metric_name=error_type,
            metric_value=1,
            metric_unit='count',
            endpoint=endpoint,
            user_id=user_id,
            metadata={'error_details': error_details}
        )
        metric.save()
        return metric
    
    @classmethod
    def get_average_response_time(cls, endpoint=None, hours=24):
        """Get average response time for endpoint"""
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        query = db.session.query(func.avg(cls.metric_value))\
                         .filter(
                             cls.metric_name == 'api_response_time',
                             cls.created_at >= datetime.utcnow() - timedelta(hours=hours)
                         )
        
        if endpoint:
            query = query.filter(cls.endpoint == endpoint)
            
        return query.scalar() or 0
    
    @classmethod
    def get_error_rate(cls, hours=24):
        """Get error rate in the last N hours"""
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        total_requests = cls.query.filter(
            cls.metric_type == 'performance',
            cls.metric_name == 'api_response_time',
            cls.created_at >= start_time
        ).count()
        
        error_count = cls.query.filter(
            cls.metric_type == 'error',
            cls.created_at >= start_time
        ).count()
        
        return (error_count / total_requests * 100) if total_requests > 0 else 0
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'endpoint': self.endpoint,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'metadata': self.metadata,
            'tags': self.tags,
            'created_at': self.created_at.isoformat()
        }

class ApiUsage(BaseModel):
    """Model for tracking API usage and rate limiting"""
    __tablename__ = 'api_usage'
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    api_key = db.Column(db.String(255), nullable=True)
    endpoint = db.Column(db.String(200), nullable=False)
    method = db.Column(db.String(10), nullable=False)  # GET, POST, etc.
    
    # Usage tracking
    request_count = db.Column(db.Integer, default=1, nullable=False)
    response_status = db.Column(db.Integer, nullable=False)
    response_time = db.Column(db.Float, nullable=False)
    
    # Request details
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    request_size = db.Column(db.Integer)  # bytes
    response_size = db.Column(db.Integer)  # bytes
    
    @classmethod
    def record_api_call(cls, endpoint, method, response_status, response_time, 
                       user_id=None, api_key=None, ip_address=None, 
                       user_agent=None, request_size=None, response_size=None):
        """Record an API call"""
        usage = cls(
            user_id=user_id,
            api_key=api_key,
            endpoint=endpoint,
            method=method,
            response_status=response_status,
            response_time=response_time,
            ip_address=ip_address,
            user_agent=user_agent,
            request_size=request_size,
            response_size=response_size
        )
        usage.save()
        return usage
    
    @classmethod
    def get_user_usage(cls, user_id, hours=24):
        """Get usage statistics for a user"""
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        usage_stats = db.session.query(
            func.count(cls.id).label('total_requests'),
            func.avg(cls.response_time).label('avg_response_time'),
            func.sum(cls.request_size).label('total_request_size'),
            func.sum(cls.response_size).label('total_response_size')
        ).filter(
            cls.user_id == user_id,
            cls.created_at >= start_time
        ).first()
        
        return {
            'total_requests': usage_stats.total_requests or 0,
            'avg_response_time': round(usage_stats.avg_response_time or 0, 2),
            'total_request_size': usage_stats.total_request_size or 0,
            'total_response_size': usage_stats.total_response_size or 0
        }
    
    def __repr__(self):
        return f'<ApiUsage {self.endpoint}: {self.response_status}>'