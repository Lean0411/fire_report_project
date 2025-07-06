"""
認證 API 端點（使用統一響應格式）
這是一個示例，展示如何使用新的 API 響應格式
"""
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import Schema, fields, validate
from services.auth.auth_service import auth_service
from data.repositories.user_repository import UserRepository
from utils.api_response import APIResponse
from utils.decorators import handle_api_errors, validate_request_json
import re

auth_v2_bp = Blueprint('auth_v2', __name__, url_prefix='/api/v2/auth')


class RegisterSchema(Schema):
    """註冊請求 Schema"""
    username = fields.Str(
        required=True,
        validate=validate.Length(min=3, max=50)
    )
    email = fields.Email(required=True)
    password = fields.Str(
        required=True,
        validate=validate.Length(min=6, max=128)
    )
    role = fields.Str(
        missing='user',
        validate=validate.OneOf(['user', 'firefighter', 'admin'])
    )


class LoginSchema(Schema):
    """登入請求 Schema"""
    username_or_email = fields.Str(required=True)
    password = fields.Str(required=True)


class ChangePasswordSchema(Schema):
    """更改密碼請求 Schema"""
    current_password = fields.Str(required=True)
    new_password = fields.Str(
        required=True,
        validate=validate.Length(min=6, max=128)
    )


class RefreshTokenSchema(Schema):
    """刷新令牌請求 Schema"""
    refresh_token = fields.Str(required=True)


def validate_password_strength(password: str) -> tuple[bool, str]:
    """驗證密碼強度"""
    if len(password) < 8:
        return False, "密碼長度至少需要 8 個字符"
    
    if not re.search(r"[A-Z]", password):
        return False, "密碼必須包含至少一個大寫字母"
    
    if not re.search(r"[a-z]", password):
        return False, "密碼必須包含至少一個小寫字母"
    
    if not re.search(r"\d", password):
        return False, "密碼必須包含至少一個數字"
    
    return True, "密碼符合要求"


@auth_v2_bp.route('/register', methods=['POST'])
@handle_api_errors
@validate_request_json(RegisterSchema)
def register():
    """註冊新用戶"""
    data = request.validated_data
    
    # 驗證密碼強度
    is_valid, message = validate_password_strength(data['password'])
    if not is_valid:
        return APIResponse.validation_error(
            errors={"password": [message]}
        )
    
    # 檢查用戶是否已存在
    user_repo = UserRepository()
    
    if user_repo.find_by_username(data['username']):
        return APIResponse.conflict(
            message="用戶名已被使用"
        )
    
    if user_repo.find_by_email(data['email']):
        return APIResponse.conflict(
            message="電子郵件已被註冊"
        )
    
    # 創建用戶
    try:
        result = auth_service.register(
            username=data['username'],
            email=data['email'],
            password=data['password'],
            role=data['role']
        )
        
        if result['success']:
            user_data = {
                'id': result['user']['id'],
                'username': result['user']['username'],
                'email': result['user']['email'],
                'role': result['user']['role']
            }
            
            return APIResponse.created(
                data={
                    'user': user_data,
                    'tokens': result['tokens']
                },
                message="註冊成功",
                location=f"/api/v2/users/{result['user']['id']}"
            )
        else:
            return APIResponse.error(
                message=result.get('message', '註冊失敗'),
                error_code="REGISTRATION_FAILED"
            )
            
    except Exception as e:
        raise  # 讓 handle_api_errors 處理


@auth_v2_bp.route('/login', methods=['POST'])
@handle_api_errors
@validate_request_json(LoginSchema)
def login():
    """用戶登入"""
    data = request.validated_data
    
    result = auth_service.login(
        username_or_email=data['username_or_email'],
        password=data['password']
    )
    
    if result['success']:
        user_data = {
            'id': result['user']['id'],
            'username': result['user']['username'],
            'email': result['user']['email'],
            'role': result['user']['role']
        }
        
        return APIResponse.success(
            data={
                'user': user_data,
                'tokens': result['tokens']
            },
            message="登入成功"
        )
    else:
        return APIResponse.unauthorized(
            message=result.get('message', '用戶名或密碼錯誤')
        )


@auth_v2_bp.route('/logout', methods=['POST'])
@jwt_required()
@handle_api_errors
def logout():
    """用戶登出"""
    jti = get_jwt()['jti']
    result = auth_service.logout(jti)
    
    if result:
        return APIResponse.success(
            message="登出成功"
        )
    else:
        return APIResponse.error(
            message="登出失敗",
            error_code="LOGOUT_FAILED"
        )


@auth_v2_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
@handle_api_errors
def refresh():
    """刷新訪問令牌"""
    identity = get_jwt_identity()
    tokens = auth_service.refresh_token(identity)
    
    return APIResponse.success(
        data={'tokens': tokens},
        message="令牌刷新成功"
    )


@auth_v2_bp.route('/change-password', methods=['PUT'])
@jwt_required()
@handle_api_errors
@validate_request_json(ChangePasswordSchema)
def change_password():
    """更改密碼"""
    user_id = get_jwt_identity()
    data = request.validated_data
    
    # 驗證新密碼強度
    is_valid, message = validate_password_strength(data['new_password'])
    if not is_valid:
        return APIResponse.validation_error(
            errors={"new_password": [message]}
        )
    
    result = auth_service.change_password(
        user_id=user_id,
        current_password=data['current_password'],
        new_password=data['new_password']
    )
    
    if result['success']:
        return APIResponse.success(
            message="密碼更改成功"
        )
    else:
        return APIResponse.error(
            message=result.get('message', '密碼更改失敗'),
            error_code="PASSWORD_CHANGE_FAILED",
            status_code=400
        )


@auth_v2_bp.route('/profile', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_profile():
    """獲取當前用戶資料"""
    user_id = get_jwt_identity()
    user_repo = UserRepository()
    user = user_repo.get_by_id(user_id)
    
    if not user:
        return APIResponse.not_found(resource="用戶")
    
    user_data = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'api_key_count': len(user.api_keys) if hasattr(user, 'api_keys') else 0
    }
    
    return APIResponse.success(
        data={'user': user_data},
        message="獲取用戶資料成功"
    )


@auth_v2_bp.route('/profile', methods=['PUT'])
@jwt_required()
@handle_api_errors
def update_profile():
    """更新用戶資料"""
    user_id = get_jwt_identity()
    
    # 定義更新 Schema
    class UpdateProfileSchema(Schema):
        email = fields.Email(required=False)
        username = fields.Str(
            required=False,
            validate=validate.Length(min=3, max=50)
        )
    
    schema = UpdateProfileSchema()
    
    try:
        data = schema.load(request.get_json())
    except ValidationError as e:
        return APIResponse.validation_error(errors=e.messages)
    
    if not data:
        return APIResponse.error(
            message="沒有提供要更新的數據",
            error_code="NO_UPDATE_DATA"
        )
    
    user_repo = UserRepository()
    
    # 檢查新的用戶名或電子郵件是否已被使用
    if 'username' in data:
        existing_user = user_repo.find_by_username(data['username'])
        if existing_user and existing_user.id != user_id:
            return APIResponse.conflict(message="用戶名已被使用")
    
    if 'email' in data:
        existing_user = user_repo.find_by_email(data['email'])
        if existing_user and existing_user.id != user_id:
            return APIResponse.conflict(message="電子郵件已被使用")
    
    # 更新用戶資料
    user = user_repo.get_by_id(user_id)
    if not user:
        return APIResponse.not_found(resource="用戶")
    
    for key, value in data.items():
        setattr(user, key, value)
    
    user_repo.update(user)
    
    return APIResponse.success(
        data={
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role
            }
        },
        message="用戶資料更新成功"
    )


@auth_v2_bp.route('/verify-token', methods=['POST'])
@jwt_required()
@handle_api_errors
def verify_token():
    """驗證當前令牌"""
    user_id = get_jwt_identity()
    jwt_data = get_jwt()
    
    return APIResponse.success(
        data={
            'valid': True,
            'user_id': user_id,
            'expires_at': jwt_data.get('exp'),
            'issued_at': jwt_data.get('iat'),
            'type': jwt_data.get('type', 'access')
        },
        message="令牌有效"
    )