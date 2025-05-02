from flask import Blueprint, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from app.config import Config
from app.utils import allowed_file, generate_timestamp, get_file_extension
from app.processor import process_image
import logging

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

@main_bp.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

@main_bp.route('/apple-touch-icon.png')
@main_bp.route('/apple-touch-icon-precomposed.png')
def apple_touch_icon():
    return send_from_directory('static', 'apple-touch-icon.png')

@main_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main_bp.route('/api/detect', methods=['POST'])
def detect_fire():
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

    # 生成唯一檔名
    timestamp = generate_timestamp()
    extension = get_file_extension(file.filename)
    filename = f"{timestamp}.{extension}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        # 儲存上傳的檔案
        file.save(filepath)
        
        # 處理圖片
        result, error = process_image(filepath, role)
        if error:
            return jsonify({'success': False, 'error': error}), 500
            
        return jsonify({
            'success': True,
            'data': {
                'filename': filename,
                **result
            }
        })
        
    except Exception as e:
        logger.error(f"處理上傳檔案時發生錯誤: {e}")
        return jsonify({'success': False, 'error': '處理檔案時發生錯誤'}), 500