# app/processor.py

import os
import json
import logging
from PIL import Image as PILImage, ImageDraw, ImageFont
from app.models import model as fire_model
from app.config import Config
from app.utils import call_gemma

logger = logging.getLogger(__name__)


def load_sop_recommendations() -> dict:
    """載入 SOP 建議內容"""
    try:
        kb_path = os.path.join(Config.BASE_DIR, 'knowledge_base', 'sop.json')
        with open(kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"無法載入 SOP 知識庫: {e}")
        return {}


def get_role_recommendations(role: str, is_fire: bool) -> dict:
    """根據角色和火災狀況取得建議"""
    sop_data = load_sop_recommendations()
    if not sop_data or role not in sop_data:
        return {}

    role_sop = sop_data[role]
    recommendations = {}

    if is_fire:
        # 火災時提供所有建議
        recommendations = role_sop
    else:
        # 否則僅提供預防性建議
        if role == "general":
            recommendations["emergency_action_plan"] = role_sop.get("emergency_action_plan", [])
        elif role == "firefighter":
            recommendations["initial_assessment"] = role_sop.get("initial_assessment", [])
        elif role == "management":
            recommendations["emergency_management_protocols"] = role_sop.get("emergency_management_protocols", [])

    return recommendations


def generate_llm_report(image_analysis: str, user_role: str) -> str:
    """
    根據影像分析結果與使用者身份呼叫 LLM，生成結構化行動建議報告。
    """
    prompt = (
        f"以下是火災初步分析：\n{image_analysis}\n\n"
        f"請扮演應急管理專家，針對「{user_role}」身份，"
        "以「狀況摘要｜優先處置｜注意事項」三個標題，"
        "用條列方式分段生成行動建議。"
    )
    return call_gemma(
        prompt,
        max_tokens=Config.LLM_MAX_TOKENS,
        temperature=Config.LLM_TEMPERATURE
    )


def draw_visualization(image_path: str, detection: dict) -> str:
    """
    在圖片左上角標註火災信心度，並存成 annotated_<filename>。
    回傳新的檔名 (basename)。
    """
    img = PILImage.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"Fire: {detection['fire_probability']}%"
    draw.text((10, 10), text, fill="red", font=font)

    dirname, fname = os.path.split(image_path)
    name, ext = os.path.splitext(fname)
    annotated_fname = f"annotated_{name}{ext}"
    annotated_path = os.path.join(dirname, annotated_fname)

    img.save(annotated_path)
    return annotated_fname


def process_image(image_path: str, role: str) -> tuple[dict, str]:
    """
    處理上傳的圖片：
      1. CNN 偵測 fire vs no-fire
      2. 產生 SOP 建議
      3. 若 fire，呼叫 LLM 生成報告
      4. 標註圖片並回傳 annotated 檔名
    回傳 (result, error_message)。若成功，error_message 為 None。
    """
    try:
        # --- CNN 偵測 ---
        with PILImage.open(image_path) as img:
            img = img.convert('RGB')
        p_fire, p_no = fire_model.predict(img)
        if p_fire is None:
            return None, "模型預測失敗"

        is_fire = p_fire > Config.CONFIDENCE_THRESHOLD
        description = (
            f"偵測到火災 (信心度: {p_fire:.1%})"
            if is_fire else
            f"未偵測到火災 (信心度: {p_no:.1%})"
        )
        detection = {
            'is_fire': is_fire,
            'description': description,
            'fire_probability': round(p_fire * 100, 1),
            'no_fire_probability': round(p_no * 100, 1)
        }

        # --- SOP 建議 ---
        recommendations = get_role_recommendations(role, is_fire)

        # --- LLM 報告 ---
        llm_report = ""
        if is_fire:
            llm_report = generate_llm_report(description, role)

        # --- 圖片標註 ---
        annotated_fname = draw_visualization(image_path, detection)

        result = {
            'detection': detection,
            'recommendations': recommendations,
            'llm_report': llm_report,
            'annotated': annotated_fname
        }
        return result, None

    except Exception as e:
        logger.error(f"圖片處理失敗: {e}")
        return None, str(e)
