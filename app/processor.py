from PIL import Image
import json
import os
from app.models import model as fire_model
from app.config import Config
import logging

logger = logging.getLogger(__name__)

def load_sop_recommendations():
    """載入 SOP 建議內容"""
    try:
        knowledge_base_path = os.path.join(Config.BASE_DIR, 'knowledge_base/sop.json')
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"無法載入 SOP 知識庫: {e}")
        return {}

def get_role_recommendations(role, is_fire):
    """根據角色和火災狀況取得建議"""
    sop_data = load_sop_recommendations()
    if not sop_data or role not in sop_data:
        return {}
    
    role_sop = sop_data[role]
    recommendations = {}
    
    if is_fire:
        # 火災情況下，提供所有相關建議
        recommendations = role_sop
    else:
        # 非火災情況下，只提供預防性建議
        if role == "general":
            recommendations["emergency_action_plan"] = role_sop["emergency_action_plan"]
        elif role == "firefighter":
            recommendations["initial_assessment"] = role_sop["initial_assessment"]
        elif role == "management":
            recommendations["emergency_management_protocols"] = role_sop["emergency_management_protocols"]
    
    return recommendations

def process_image(image_path, role):
    """處理上傳的圖片並生成分析結果"""
    try:
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            # 進行火災偵測
            p_fire, p_no = fire_model.predict(image)
            if p_fire is None:
                return None, "模型預測失敗"
            
            is_fire = p_fire > Config.CONFIDENCE_THRESHOLD
            
            # 根據角色獲取建議
            recommendations = get_role_recommendations(role, is_fire)
            
            result = {
                'detection': {
                    'is_fire': is_fire,
                    'description': f'偵測到火災 (信心度: {p_fire:.1%})' if is_fire else f'未偵測到火災 (信心度: {p_no:.1%})',
                    'fire_probability': round(p_fire * 100, 1),
                    'no_fire_probability': round(p_no * 100, 1)
                },
                'recommendations': recommendations
            }
            
            return result, None
            
    except Exception as e:
        logger.error(f"圖片處理失敗: {e}")
        return None, str(e)

def draw_visualization(image_path, detection_result):
    """在圖片上繪製視覺化結果（如需要）"""
    # TODO: 實作視覺化邏輯，例如在圖片上標註信心度等
    pass