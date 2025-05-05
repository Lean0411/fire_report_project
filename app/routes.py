# app/routes.py

import os
import logging
from flask import Blueprint, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from app.config import Config
from app.utils import allowed_file, generate_timestamp, get_file_extension
from app.processor import process_image

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

# 分類標籤對應表，供模板顯示中文
category_labels = {
    'emergency_action_plan': '緊急應變計畫',
    'evacuation_preparedness': '疏散準備',
    'shelter_in_place': '就地避難',
    'initial_assessment': '初步評估',
    'incident_command': '事件指揮',
    'tactical_operations': '戰術行動',
    'personnel_safety': '人員安全',
    'emergency_management_protocols': '緊急管理協定',
    'resource_allocation': '資源分配',
    'communication_coordination': '通訊協調',
    'resilience_and_recovery': '韌性與復原'
}


@main_bp.route('/favicon.ico')
def favicon():
    """提供 favicon"""
    return send_from_directory('static', 'favicon.ico')


@main_bp.route('/apple-touch-icon.png')
@main_bp.route('/apple-touch-icon-precomposed.png')
def apple_touch_icon():
    """提供 Apple Touch Icon"""
    return send_from_directory('static', 'apple-touch-icon.png')


@main_bp.route('/', methods=['GET'])
def index():
    """
    首頁路由：渲染 index.html，
    並注入 category_labels 讓模板內可正確顯示建議分類名稱。
    """
    return render_template('index.html', category_labels=category_labels)


@main_bp.route('/api/detect', methods=['POST'])
def detect_fire():
    """
    API：接收上傳影像與身份參數，執行火災偵測流程，
    回傳 JSON 格式的偵測結果、SOP 建議，以及 LLM 報告。
    """
    # 1. 接收並驗證檔案與身份
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未收到檔案'}), 400

    role = request.form.get('role', 'general')
    if role not in ['general', 'firefighter', 'management']:
        return jsonify({'success': False, 'error': '無效的身份類型'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '未選擇檔案'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '檔案格式不支援'}), 400

    # 2. 生成唯一檔名並儲存
    timestamp = generate_timestamp()
    ext = get_file_extension(file.filename)
    filename = f"{timestamp}.{ext}"
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    try:
        file.save(save_path)
    except Exception as e:
        logger.error(f"檔案儲存失敗: {e}")
        return jsonify({'success': False, 'error': '檔案儲存失敗'}), 500

    # 3. 處理圖片（CNN 偵測 + SOP 建議 + LLM 報告 + 標註圖產生）
    result, error = process_image(save_path, role)
    if error:
        return jsonify({'success': False, 'error': error}), 500

    # 4. 回傳 JSON
    return jsonify({
        'success': True,
        'data': {
            'filename': filename,
            'detection': result['detection'],
            'recommendations': result['recommendations'],
            'llm_report': result.get('llm_report', '')
        }
    })
