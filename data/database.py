from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
from typing import Dict, Any
import os

db = SQLAlchemy()
migrate = Migrate()

def init_db(app: Flask) -> SQLAlchemy:
    """Initialize database with Flask app"""
    # Database configuration
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or \
        f'sqlite:///{os.path.join(basedir, "../fire_guard.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    migrate.init_app(app, db)
    
    return db

class BaseModel(db.Model):
    """Base model with common fields"""
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    def save(self) -> None:
        """Save model to database"""
        db.session.add(self)
        db.session.commit()
        return self
    
    def delete(self) -> None:
        """Delete model from database"""
        db.session.delete(self)
        db.session.commit()