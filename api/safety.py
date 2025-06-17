"""
安全建議API藍圖
提供火災安全建議相關的API端點
"""
from flask import Blueprint, request, jsonify

from services.sop_service import sop_service
from services.safety_service import safety_service
from utils.constants import HTTP_STATUS, USER_ROLES
from config.logging_config import get_logger

logger = get_logger(__name__)

# 創建藍圖
safety_bp = Blueprint('safety', __name__)

@safety_bp.route('/api/fire-safety-advice', methods=['GET', 'POST'])
def get_fire_safety_advice():
    """
    獲取火災安全建議API
    
    支援GET和POST請求，可指定角色獲取對應建議
    """
    try:
        # 獲取角色參數
        if request.method == 'POST':
            if request.is_json:
                role = request.json.get('role', 'general')
            else:
                role = request.form.get('role', 'general')
        else:
            role = request.args.get('role', 'general')
        
        # 驗證角色
        if role not in USER_ROLES:
            logger.warning(f"請求了無效角色: {role}")
            role = 'general'  # 預設為一般民眾
        
        # 獲取該角色的所有建議
        recommendations = sop_service.get_all_recommendations_for_role(role)
        
        if not recommendations:
            # 如果SOP數據為空，使用預設安全建議
            logger.info(f"SOP數據為空，使用預設安全建議為角色: {role}")
            recommendations = safety_service.generate_fire_safety_tips()
        
        logger.info(f"成功提供角色 '{role}' 的安全建議")
        
        return jsonify(recommendations), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取安全建議時發生錯誤: {e}")
        return jsonify({
            'error': '獲取安全建議失敗，請稍後再試'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

@safety_bp.route('/api/safety/general-tips', methods=['GET'])
def get_general_safety_tips():
    """獲取一般火災安全建議"""
    try:
        tips = safety_service.generate_fire_safety_tips()
        
        return jsonify({
            'success': True,
            'data': tips
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取一般安全建議失敗: {e}")
        return jsonify({
            'success': False,
            'error': '獲取安全建議失敗'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

@safety_bp.route('/api/safety/situation/<situation>', methods=['GET'])
def get_situation_advice(situation):
    """
    獲取特定情況的安全建議
    
    Args:
        situation: 情況類型 (immediate_actions/evacuation_tips/prevention_measures等)
    """
    try:
        advice = safety_service.get_situation_specific_advice(situation)
        
        if not advice:
            return jsonify({
                'success': False,
                'error': f'未找到情況 "{situation}" 的建議'
            }), HTTP_STATUS['BAD_REQUEST']
        
        return jsonify({
            'success': True,
            'situation': situation,
            'advice': advice
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取情況建議失敗: {e}")
        return jsonify({
            'success': False,
            'error': '獲取建議失敗'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

@safety_bp.route('/api/safety/emergency-contacts', methods=['GET'])
def get_emergency_contacts():
    """獲取緊急聯絡方式"""
    try:
        contacts = safety_service.get_emergency_contacts()
        
        return jsonify({
            'success': True,
            'data': contacts
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取緊急聯絡方式失敗: {e}")
        return jsonify({
            'success': False,
            'error': '獲取緊急聯絡方式失敗'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

@safety_bp.route('/api/safety/checklist', methods=['GET'])
def get_safety_checklist():
    """
    獲取安全檢查清單
    
    Query參數:
        environment: 環境類型 (general/home/office/factory)
    """
    try:
        environment = request.args.get('environment', 'general')
        checklist = safety_service.get_safety_checklist(environment)
        
        return jsonify({
            'success': True,
            'environment': environment,
            'checklist': checklist
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取安全檢查清單失敗: {e}")
        return jsonify({
            'success': False,
            'error': '獲取檢查清單失敗'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

@safety_bp.route('/api/safety/role-advice', methods=['POST'])
def get_role_based_advice():
    """
    獲取基於角色的個性化安全建議
    
    請求體:
        role: 使用者角色
        is_fire: 是否檢測到火災 (可選)
        fire_probability: 火災機率 (可選)
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '請求必須為JSON格式'
            }), HTTP_STATUS['BAD_REQUEST']
        
        data = request.json
        role = data.get('role')
        is_fire = data.get('is_fire', False)
        fire_probability = data.get('fire_probability', 0.0)
        
        # 驗證角色
        if not role or role not in USER_ROLES:
            return jsonify({
                'success': False,
                'error': f'無效的角色，支援的角色: {list(USER_ROLES.keys())}'
            }), HTTP_STATUS['BAD_REQUEST']
        
        # 生成個性化建議
        advice = safety_service.generate_role_based_advice(
            role, is_fire, fire_probability
        )
        
        return jsonify({
            'success': True,
            'data': advice
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取角色建議失敗: {e}")
        return jsonify({
            'success': False,
            'error': '獲取個性化建議失敗'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

@safety_bp.route('/api/safety/sop/validate', methods=['GET'])
def validate_sop_data():
    """驗證SOP數據完整性"""
    try:
        validation_result = sop_service.validate_sop_data()
        
        return jsonify({
            'success': True,
            'data': validation_result
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"SOP數據驗證失敗: {e}")
        return jsonify({
            'success': False,
            'error': 'SOP數據驗證失敗'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

@safety_bp.route('/api/safety/roles', methods=['GET'])
def get_available_roles():
    """獲取可用的角色列表"""
    try:
        roles = sop_service.get_available_roles()
        
        return jsonify({
            'success': True,
            'data': {
                'roles': roles,
                'total': len(roles)
            }
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取角色列表失敗: {e}")
        return jsonify({
            'success': False,
            'error': '獲取角色列表失敗'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']