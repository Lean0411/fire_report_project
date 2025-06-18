from typing import Optional, List
from .base_repository import BaseRepository
from ..models.user_model import User

class UserRepository(BaseRepository):
    """Repository for User model operations"""
    
    def __init__(self):
        super().__init__(User)
    
    def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        return self.find_one_by(username=username)
    
    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email"""
        return self.find_one_by(email=email)
    
    def find_by_api_key(self, api_key: str) -> Optional[User]:
        """Find user by API key"""
        return self.find_one_by(api_key=api_key)
    
    def authenticate(self, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password"""
        # Try to find by username first
        user = self.find_by_username(username_or_email)
        
        # If not found, try email
        if not user:
            user = self.find_by_email(username_or_email)
        
        # Check password if user found
        if user and user.check_password(password) and user.is_active:
            user.update_last_login()
            return user
        
        return None
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> Optional[User]:
        """Create a new user"""
        # Check if username or email already exists
        if self.find_by_username(username):
            raise ValueError(f"Username '{username}' already exists")
        
        if self.find_by_email(email):
            raise ValueError(f"Email '{email}' already exists")
        
        return self.create(
            username=username,
            email=email,
            password=password,
            role=role
        )
    
    def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        user = self.get_by_id(user_id)
        if user:
            user.set_password(new_password)
            from ..database import db
            db.session.commit()
            return True
        return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account"""
        return self.update(user_id, is_active=False) is not None
    
    def activate_user(self, user_id: int) -> bool:
        """Activate user account"""
        return self.update(user_id, is_active=True) is not None
    
    def get_active_users(self) -> List[User]:
        """Get all active users"""
        return self.find_by(is_active=True)
    
    def get_users_by_role(self, role: str) -> List[User]:
        """Get users by role"""
        return self.find_by(role=role, is_active=True)
    
    def regenerate_api_key(self, user_id: int) -> Optional[str]:
        """Regenerate API key for user"""
        user = self.get_by_id(user_id)
        if user:
            new_api_key = user.generate_api_key()
            from ..database import db
            db.session.commit()
            return new_api_key
        return None
    
    def search_users(self, query: str, limit: int = 50) -> List[User]:
        """Search users by username or email"""
        return User.query.filter(
            (User.username.contains(query)) | 
            (User.email.contains(query))
        ).filter(User.is_active == True).limit(limit).all()