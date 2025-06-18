from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import Text
from ..database import db, BaseModel

class DetectionHistory(BaseModel):
    """Model for storing fire detection analysis history"""
    __tablename__ = 'detection_history'
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    image_filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    
    # Detection results
    fire_detected = db.Column(db.Boolean, nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    detection_method = db.Column(db.String(50), nullable=False)  # 'cnn', 'ai_analysis', 'combined'
    
    # AI Analysis results
    ai_analysis = db.Column(Text)  # Full AI analysis text
    risk_level = db.Column(db.String(20))  # low, medium, high, critical
    fire_type = db.Column(db.String(100))  # Type of fire detected
    location_description = db.Column(db.String(500))  # Where the fire is located
    
    # Recommendations
    recommendations = db.Column(JSON)  # JSON array of recommendations
    safety_instructions = db.Column(Text)  # Safety instructions text
    
    # User context
    user_role = db.Column(db.String(50))  # Role when detection was made
    user_location = db.Column(db.String(200))  # User's location if provided
    
    # Processing metadata
    processing_time = db.Column(db.Float)  # Time taken to process (seconds)
    model_version = db.Column(db.String(50))  # Version of model used
    ai_provider = db.Column(db.String(50))  # openai, ollama, etc.
    
    # Additional metadata
    image_metadata = db.Column(JSON)  # Image dimensions, format, etc.
    session_id = db.Column(db.String(100))  # Session identifier
    ip_address = db.Column(db.String(45))  # Client IP address
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self, include_full_analysis=True):
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'image_filename': self.image_filename,
            'fire_detected': self.fire_detected,
            'confidence_score': self.confidence_score,
            'detection_method': self.detection_method,
            'risk_level': self.risk_level,
            'fire_type': self.fire_type,
            'location_description': self.location_description,
            'recommendations': self.recommendations,
            'user_role': self.user_role,
            'user_location': self.user_location,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'ai_provider': self.ai_provider,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if include_full_analysis:
            data['ai_analysis'] = self.ai_analysis
            data['safety_instructions'] = self.safety_instructions
            
        return data
    
    def get_summary(self):
        """Get a summary of the detection"""
        return {
            'id': self.id,
            'fire_detected': self.fire_detected,
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level,
            'fire_type': self.fire_type,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def get_user_history(cls, user_id, limit=50):
        """Get detection history for a user"""
        return cls.query.filter_by(user_id=user_id)\
                      .order_by(cls.created_at.desc())\
                      .limit(limit).all()
    
    @classmethod
    def get_recent_detections(cls, hours=24, fire_only=False):
        """Get recent detections"""
        from datetime import datetime, timedelta
        
        query = cls.query.filter(
            cls.created_at >= datetime.utcnow() - timedelta(hours=hours)
        )
        
        if fire_only:
            query = query.filter(cls.fire_detected == True)
            
        return query.order_by(cls.created_at.desc()).all()
    
    @classmethod
    def get_statistics(cls, days=30):
        """Get detection statistics"""
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        total_detections = cls.query.filter(cls.created_at >= start_date).count()
        fire_detections = cls.query.filter(
            cls.created_at >= start_date,
            cls.fire_detected == True
        ).count()
        
        avg_confidence = db.session.query(func.avg(cls.confidence_score))\
                                  .filter(cls.created_at >= start_date)\
                                  .scalar() or 0
        
        risk_distribution = db.session.query(
            cls.risk_level,
            func.count(cls.risk_level)
        ).filter(
            cls.created_at >= start_date,
            cls.fire_detected == True
        ).group_by(cls.risk_level).all()
        
        return {
            'total_detections': total_detections,
            'fire_detections': fire_detections,
            'fire_rate': (fire_detections / total_detections * 100) if total_detections > 0 else 0,
            'average_confidence': round(avg_confidence, 2),
            'risk_distribution': dict(risk_distribution)
        }
    
    def __repr__(self):
        return f'<DetectionHistory {self.id}: {self.fire_detected}>'