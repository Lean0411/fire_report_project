from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy import desc, asc, and_, or_
from .base_repository import BaseRepository
from ..models.detection_history import DetectionHistory

class DetectionRepository(BaseRepository):
    """Repository for DetectionHistory model operations"""
    
    def __init__(self):
        super().__init__(DetectionHistory)
    
    def create_detection(self, **kwargs) -> DetectionHistory:
        """Create a new detection record"""
        return self.create(**kwargs)
    
    def get_user_detections(self, user_id: int, limit: int = 50, offset: int = 0) -> List[DetectionHistory]:
        """Get detections for a specific user"""
        return DetectionHistory.query.filter_by(user_id=user_id)\
                                   .order_by(desc(DetectionHistory.created_at))\
                                   .limit(limit)\
                                   .offset(offset)\
                                   .all()
    
    def get_recent_detections(self, hours: int = 24, fire_only: bool = False, limit: int = 100) -> List[DetectionHistory]:
        """Get recent detections"""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = DetectionHistory.query.filter(DetectionHistory.created_at >= start_time)
        
        if fire_only:
            query = query.filter(DetectionHistory.fire_detected == True)
        
        return query.order_by(desc(DetectionHistory.created_at)).limit(limit).all()
    
    def get_detections_by_date_range(self, start_date: datetime, end_date: datetime, 
                                   user_id: Optional[int] = None) -> List[DetectionHistory]:
        """Get detections within a date range"""
        query = DetectionHistory.query.filter(
            and_(
                DetectionHistory.created_at >= start_date,
                DetectionHistory.created_at <= end_date
            )
        )
        
        if user_id:
            query = query.filter(DetectionHistory.user_id == user_id)
        
        return query.order_by(desc(DetectionHistory.created_at)).all()
    
    def get_fire_detections_by_risk_level(self, risk_level: str, limit: int = 50) -> List[DetectionHistory]:
        """Get fire detections by risk level"""
        return DetectionHistory.query.filter(
            and_(
                DetectionHistory.fire_detected == True,
                DetectionHistory.risk_level == risk_level
            )
        ).order_by(desc(DetectionHistory.created_at)).limit(limit).all()
    
    def search_detections(self, 
                         fire_detected: Optional[bool] = None,
                         risk_level: Optional[str] = None,
                         fire_type: Optional[str] = None,
                         user_role: Optional[str] = None,
                         min_confidence: Optional[float] = None,
                         max_confidence: Optional[float] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         limit: int = 100) -> List[DetectionHistory]:
        """Advanced search for detections"""
        query = DetectionHistory.query
        
        if fire_detected is not None:
            query = query.filter(DetectionHistory.fire_detected == fire_detected)
        
        if risk_level:
            query = query.filter(DetectionHistory.risk_level == risk_level)
        
        if fire_type:
            query = query.filter(DetectionHistory.fire_type.contains(fire_type))
        
        if user_role:
            query = query.filter(DetectionHistory.user_role == user_role)
        
        if min_confidence is not None:
            query = query.filter(DetectionHistory.confidence_score >= min_confidence)
        
        if max_confidence is not None:
            query = query.filter(DetectionHistory.confidence_score <= max_confidence)
        
        if start_date:
            query = query.filter(DetectionHistory.created_at >= start_date)
        
        if end_date:
            query = query.filter(DetectionHistory.created_at <= end_date)
        
        return query.order_by(desc(DetectionHistory.created_at)).limit(limit).all()
    
    def get_detection_statistics(self, days: int = 30, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        from sqlalchemy import func
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        base_query = DetectionHistory.query.filter(DetectionHistory.created_at >= start_date)
        
        if user_id:
            base_query = base_query.filter(DetectionHistory.user_id == user_id)
        
        # Basic counts
        total_detections = base_query.count()
        fire_detections = base_query.filter(DetectionHistory.fire_detected == True).count()
        
        # Confidence statistics
        confidence_stats = base_query.with_entities(
            func.avg(DetectionHistory.confidence_score).label('avg_confidence'),
            func.min(DetectionHistory.confidence_score).label('min_confidence'),
            func.max(DetectionHistory.confidence_score).label('max_confidence')
        ).first()
        
        # Risk level distribution
        risk_distribution = base_query.filter(DetectionHistory.fire_detected == True)\
                                    .with_entities(
                                        DetectionHistory.risk_level,
                                        func.count(DetectionHistory.risk_level).label('count')
                                    )\
                                    .group_by(DetectionHistory.risk_level)\
                                    .all()
        
        # Fire type distribution
        fire_type_distribution = base_query.filter(DetectionHistory.fire_detected == True)\
                                         .with_entities(
                                             DetectionHistory.fire_type,
                                             func.count(DetectionHistory.fire_type).label('count')
                                         )\
                                         .group_by(DetectionHistory.fire_type)\
                                         .all()
        
        # Detection method distribution
        method_distribution = base_query.with_entities(
            DetectionHistory.detection_method,
            func.count(DetectionHistory.detection_method).label('count')
        ).group_by(DetectionHistory.detection_method).all()
        
        # Average processing time
        avg_processing_time = base_query.with_entities(
            func.avg(DetectionHistory.processing_time)
        ).scalar() or 0
        
        return {
            'period_days': days,
            'total_detections': total_detections,
            'fire_detections': fire_detections,
            'fire_rate': (fire_detections / total_detections * 100) if total_detections > 0 else 0,
            'confidence_stats': {
                'average': round(confidence_stats.avg_confidence or 0, 2),
                'minimum': round(confidence_stats.min_confidence or 0, 2),
                'maximum': round(confidence_stats.max_confidence or 0, 2)
            },
            'risk_distribution': {level: count for level, count in risk_distribution},
            'fire_type_distribution': {fire_type: count for fire_type, count in fire_type_distribution if fire_type},
            'detection_method_distribution': {method: count for method, count in method_distribution},
            'average_processing_time': round(avg_processing_time, 3)
        }
    
    def get_daily_detection_counts(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily detection counts for the last N days"""
        from sqlalchemy import func, cast, Date
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        daily_counts = DetectionHistory.query.filter(DetectionHistory.created_at >= start_date)\
                                           .with_entities(
                                               cast(DetectionHistory.created_at, Date).label('date'),
                                               func.count(DetectionHistory.id).label('total_detections'),
                                               func.sum(func.cast(DetectionHistory.fire_detected, int)).label('fire_detections')
                                           )\
                                           .group_by(cast(DetectionHistory.created_at, Date))\
                                           .order_by(cast(DetectionHistory.created_at, Date))\
                                           .all()
        
        return [
            {
                'date': str(date),
                'total_detections': total,
                'fire_detections': fire_count or 0,
                'fire_rate': (fire_count / total * 100) if total > 0 and fire_count else 0
            }
            for date, total, fire_count in daily_counts
        ]
    
    def delete_old_detections(self, days: int = 365) -> int:
        """Delete detection records older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        old_detections = DetectionHistory.query.filter(DetectionHistory.created_at < cutoff_date)
        count = old_detections.count()
        old_detections.delete()
        
        from ..database import db
        db.session.commit()
        
        return count