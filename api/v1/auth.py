from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import Schema, fields, ValidationError
from services.auth.auth_service import auth_service
from data.repositories.user_repository import UserRepository
import re

auth_bp = Blueprint('auth_v1', __name__, url_prefix='/api/v1/auth')

class RegisterSchema(Schema):
    username = fields.Str(required=True, validate=lambda x: len(x) >= 3)
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=lambda x: len(x) >= 6)
    role = fields.Str(missing='user', validate=lambda x: x in ['user', 'firefighter', 'admin'])

class LoginSchema(Schema):
    username_or_email = fields.Str(required=True)
    password = fields.Str(required=True)

class ChangePasswordSchema(Schema):
    current_password = fields.Str(required=True)
    new_password = fields.Str(required=True, validate=lambda x: len(x) >= 6)

def validate_password_strength(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit"
    
    return True, "Password is valid"

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    schema = RegisterSchema()
    
    try:
        data = schema.load(request.json)
    except ValidationError as err:
        return jsonify({'error': 'Validation error', 'details': err.messages}), 400
    
    # Validate password strength
    is_valid, message = validate_password_strength(data['password'])
    if not is_valid:
        return jsonify({'error': message}), 400
    
    try:
        user = auth_service.create_user(
            username=data['username'],
            email=data['email'],
            password=data['password'],
            role=data['role']
        )
        
        # Generate tokens
        tokens = auth_service.generate_tokens(user)
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'tokens': tokens
        }), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login"""
    schema = LoginSchema()
    
    try:
        data = schema.load(request.json)
    except ValidationError as err:
        return jsonify({'error': 'Validation error', 'details': err.messages}), 400
    
    user = auth_service.authenticate_user(
        data['username_or_email'],
        data['password']
    )
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is deactivated'}), 403
    
    # Generate tokens
    tokens = auth_service.generate_tokens(user)
    
    # Update last login
    user.update_last_login()
    
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(),
        'tokens': tokens
    }), 200

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    current_user_id = get_jwt_identity()
    user_repo = UserRepository()
    user = user_repo.get_by_id(current_user_id)
    
    if not user or not user.is_active:
        return jsonify({'error': 'User not found or deactivated'}), 404
    
    # Generate new access token
    tokens = auth_service.generate_tokens(user)
    
    return jsonify({
        'message': 'Token refreshed successfully',
        'tokens': tokens
    }), 200

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """User logout (token blacklisting would be implemented here)"""
    # In a production environment, you would typically blacklist the token
    # For now, we'll just return a success message
    return jsonify({'message': 'Logged out successfully'}), 200

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    current_user_id = get_jwt_identity()
    user_repo = UserRepository()
    user = user_repo.get_by_id(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user': user.to_dict(include_sensitive=True)
    }), 200

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update current user profile"""
    current_user_id = get_jwt_identity()
    user_repo = UserRepository()
    user = user_repo.get_by_id(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.json
    
    # Update allowed fields
    if 'email' in data:
        # Check if email is already taken by another user
        existing_user = user_repo.find_by_email(data['email'])
        if existing_user and existing_user.id != user.id:
            return jsonify({'error': 'Email already exists'}), 400
        user.email = data['email']
    
    try:
        user.save()
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
    except Exception as e:
        current_app.logger.error(f"Profile update error: {str(e)}")
        return jsonify({'error': 'Profile update failed'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change user password"""
    schema = ChangePasswordSchema()
    
    try:
        data = schema.load(request.json)
    except ValidationError as err:
        return jsonify({'error': 'Validation error', 'details': err.messages}), 400
    
    current_user_id = get_jwt_identity()
    user_repo = UserRepository()
    user = user_repo.get_by_id(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Verify current password
    if not user.check_password(data['current_password']):
        return jsonify({'error': 'Current password is incorrect'}), 400
    
    # Validate new password strength
    is_valid, message = validate_password_strength(data['new_password'])
    if not is_valid:
        return jsonify({'error': message}), 400
    
    # Update password
    try:
        user_repo.update_password(user.id, data['new_password'])
        return jsonify({'message': 'Password changed successfully'}), 200
    except Exception as e:
        current_app.logger.error(f"Password change error: {str(e)}")
        return jsonify({'error': 'Password change failed'}), 500

@auth_bp.route('/regenerate-api-key', methods=['POST'])
@jwt_required()
def regenerate_api_key():
    """Regenerate API key for current user"""
    current_user_id = get_jwt_identity()
    user_repo = UserRepository()
    
    try:
        new_api_key = user_repo.regenerate_api_key(current_user_id)
        if new_api_key:
            return jsonify({
                'message': 'API key regenerated successfully',
                'api_key': new_api_key
            }), 200
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        current_app.logger.error(f"API key regeneration error: {str(e)}")
        return jsonify({'error': 'API key regeneration failed'}), 500

@auth_bp.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    """Verify if current token is valid"""
    current_user_id = get_jwt_identity()
    claims = get_jwt()
    
    return jsonify({
        'valid': True,
        'user_id': current_user_id,
        'role': claims.get('role'),
        'username': claims.get('username')
    }), 200