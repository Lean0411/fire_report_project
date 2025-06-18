from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from ..database import db

class BaseRepository(ABC):
    """Base repository class with common database operations"""
    
    def __init__(self, model):
        self.model = model
    
    def create(self, **kwargs) -> Optional[Any]:
        """Create a new record"""
        try:
            instance = self.model(**kwargs)
            db.session.add(instance)
            db.session.commit()
            return instance
        except SQLAlchemyError as e:
            db.session.rollback()
            raise e
    
    def get_by_id(self, id: int) -> Optional[Any]:
        """Get record by ID"""
        return self.model.query.get(id)
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Any]:
        """Get all records with optional pagination"""
        query = self.model.query
        
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
            
        return query.all()
    
    def update(self, id: int, **kwargs) -> Optional[Any]:
        """Update record by ID"""
        try:
            instance = self.get_by_id(id)
            if not instance:
                return None
            
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            db.session.commit()
            return instance
        except SQLAlchemyError as e:
            db.session.rollback()
            raise e
    
    def delete(self, id: int) -> bool:
        """Delete record by ID"""
        try:
            instance = self.get_by_id(id)
            if not instance:
                return False
            
            db.session.delete(instance)
            db.session.commit()
            return True
        except SQLAlchemyError as e:
            db.session.rollback()
            raise e
    
    def find_by(self, **kwargs) -> List[Any]:
        """Find records by criteria"""
        return self.model.query.filter_by(**kwargs).all()
    
    def find_one_by(self, **kwargs) -> Optional[Any]:
        """Find one record by criteria"""
        return self.model.query.filter_by(**kwargs).first()
    
    def count(self, **kwargs) -> int:
        """Count records matching criteria"""
        if kwargs:
            return self.model.query.filter_by(**kwargs).count()
        return self.model.query.count()
    
    def exists(self, **kwargs) -> bool:
        """Check if record exists"""
        return self.model.query.filter_by(**kwargs).first() is not None