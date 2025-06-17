"""
火災檢測API藍圖
提供火災檢測相關的API端點
"""
import os
import time
from flask import Blueprint, request, jsonify, current_app
from werkzeug.datastructures import FileStorage

from models.model_utils import model_manager
from services.ai_service import ai_service
from services.image_service import image_service
from services.sop_service import sop_service
from utils.file_utils import secure_save_file
from utils.constants import HTTP_STATUS, AI_PROVIDERS, USER_ROLES
from config.logging_config import get_logger

logger = get_logger(__name__)

# 創建藍圖
detection_bp = Blueprint('detection', __name__)

@detection_bp.route('/api/detect', methods=['POST'])
def detect_fire():
    """
    火災檢測主API
    
    接收圖片和參數，進行火災檢測並返回結果
    """
    start_time = time.time()
    
    try:
        # 檢查是否有檔案上傳
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '未找到上傳檔案'
            }), HTTP_STATUS['BAD_REQUEST']
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未選擇檔案'
            }), HTTP_STATUS['BAD_REQUEST']
        
        # 獲取參數
        role = request.form.get('role', '').strip()
        use_ai = request.form.get('use_ai', 'false').lower() == 'true'
        ai_provider = request.form.get('ai_provider', 'openai').lower()
        
        # 驗證角色
        if not role or role not in USER_ROLES:
            return jsonify({
                'success': False,
                'error': f'無效的角色，支援的角色: {list(USER_ROLES.keys())}'
            }), HTTP_STATUS['BAD_REQUEST']
        
        # 驗證AI提供者
        if use_ai and ai_provider not in AI_PROVIDERS:
            return jsonify({
                'success': False,
                'error': f'無效的AI提供者，支援: {list(AI_PROVIDERS.keys())}'
            }), HTTP_STATUS['BAD_REQUEST']
        
        # 保存上傳檔案
        upload_success, file_path, error_msg = secure_save_file(
            file, current_app.config['UPLOAD_FOLDER']
        )
        
        if not upload_success:
            return jsonify({
                'success': False,
                'error': error_msg
            }), HTTP_STATUS['BAD_REQUEST']
        
        # 進行火災檢測
        prediction_result = model_manager.predict_fire(file_path)
        if prediction_result is None:
            return jsonify({
                'success': False,
                'error': '模型預測失敗，請檢查模型是否正確載入'
            }), HTTP_STATUS['INTERNAL_SERVER_ERROR']
        
        p_fire, p_no_fire = prediction_result
        is_fire = p_fire > 0.5
        
        logger.info(f"火災檢測完成 - 火災機率: {p_fire:.3f}, 角色: {role}")
        
        # 生成分析結果圖片
        filename = os.path.basename(file_path)
        annotated_filename = image_service.generate_annotated_image(
            file_path, filename, is_fire, p_fire, p_no_fire
        )
        
        # 獲取SOP建議
        recommendations = sop_service.get_role_recommendations(role, is_fire)
        
        # 構建基本回應
        response_data = {
            'detection': {
                'is_fire': is_fire,
                'fire_probability': round(p_fire * 100, 1),
                'no_fire_probability': round(p_no_fire * 100, 1),
                'model_confidence': round(max(p_fire, p_no_fire), 3)
            },
            'filename': annotated_filename,
            'recommendations': recommendations,
            'processing_time': round(time.time() - start_time, 2)
        }
        
        # 如果啟用AI分析
        if use_ai:
            ai_analysis = _get_ai_analysis(file_path, role, is_fire, p_fire, ai_provider)
            if ai_analysis:
                response_data['llm_report'] = ai_analysis
        
        return jsonify({
            'success': True,
            'data': response_data
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"火災檢測API發生錯誤: {e}")
        return jsonify({
            'success': False,
            'error': '系統內部錯誤，請稍後再試'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']

def _get_ai_analysis(image_path: str, role: str, is_fire: bool, 
                     fire_probability: float, ai_provider: str) -> str:
    """
    獲取AI分析結果
    
    Args:
        image_path: 圖片路徑
        role: 使用者角色
        is_fire: 是否檢測到火災
        fire_probability: 火災機率
        ai_provider: AI提供者
        
    Returns:
        str: AI分析結果
    """
    try:
        # 構建提示詞
        role_name = USER_ROLES.get(role, '使用者')
        fire_status = "檢測到火災" if is_fire else "未檢測到明顯火災"
        
        prompt = f"""請以專業消防安全專家的角度分析這張圖片：

1. 圖片基本情況分析
2. 潛在火災風險評估  
3. 針對{role_name}的專業建議

檢測結果：{fire_status}（信心度：{fire_probability:.1%}）

請提供實用、具體的專業建議，避免一般性說明。"""
        
        # 根據AI提供者調用對應服務
        if ai_provider == 'openai':
            return ai_service.call_openai_gpt(prompt, image_path=image_path)
        elif ai_provider == 'ollama':
            return ai_service.call_ollama_gemma(prompt, image_path=image_path)
        else:
            logger.warning(f"不支援的AI提供者: {ai_provider}")
            return ""
            
    except Exception as e:
        logger.error(f"AI分析失敗: {e}")
        return ""

@detection_bp.route('/api/detect/status', methods=['GET'])
def get_detection_status():
    """獲取檢測系統狀態"""
    try:
        model_status = model_manager.get_model_status()
        device_info = model_manager.get_device_info()
        ai_status = ai_service.get_service_status()
        
        return jsonify({
            'success': True,
            'data': {
                'model': model_status,
                'device': device_info,
                'ai_services': ai_status,
                'supported_roles': USER_ROLES,
                'supported_ai_providers': AI_PROVIDERS
            }
        }), HTTP_STATUS['OK']
        
    except Exception as e:
        logger.error(f"獲取系統狀態失敗: {e}")
        return jsonify({
            'success': False,
            'error': '無法獲取系統狀態'
        }), HTTP_STATUS['INTERNAL_SERVER_ERROR']