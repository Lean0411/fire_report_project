from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
import secrets
from ..database import db, BaseModel

class User(BaseModel):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default='user', nullable=False)  # admin, firefighter, user
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_login = db.Column(db.DateTime)
    api_key = db.Column(db.String(255), unique=True)
    
    # Relationships
    detection_history = db.relationship('DetectionHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, username, email, password, role='user'):
        self.username = username
        self.email = email
        self.set_password(password)
        self.role = role
        self.generate_api_key()
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def generate_api_key(self):
        """Generate API key for the user"""
        self.api_key = secrets.token_urlsafe(32)
        return self.api_key
    
    def get_access_token(self, expires_delta=None):
        """Generate JWT access token"""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)
        
        additional_claims = {
            'role': self.role,
            'username': self.username
        }
        
        return create_access_token(
            identity=self.id,
            expires_delta=expires_delta,
            additional_claims=additional_claims
        )
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def has_role(self, role):
        """Check if user has specific role"""
        return self.role == role
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'
    
    def is_firefighter(self):
        """Check if user is firefighter"""
        return self.role == 'firefighter'
    
    def to_dict(self, include_sensitive=False):
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if include_sensitive:
            data['api_key'] = self.api_key
            
        return data
    
    def __repr__(self):
        return f'<User {self.username}>'