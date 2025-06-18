from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, ValidationError
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime

from services.auth.auth_service import auth_service
from services.ai_service import AIService
from data.repositories.detection_repository import DetectionRepository
from data.repositories.user_repository import UserRepository
from utils.image_utils import process_uploaded_image, validate_image

detection_bp = Blueprint('detection_v1', __name__, url_prefix='/api/v1/detection')

class DetectionRequestSchema(Schema):
    user_role = fields.Str(missing='user', validate=lambda x: x in ['user', 'firefighter', 'admin'])
    user_location = fields.Str(missing=None)
    use_ai_analysis = fields.Bool(missing=True)

class DetectionHistorySchema(Schema):
    page = fields.Int(missing=1, validate=lambda x: x > 0)
    per_page = fields.Int(missing=20, validate=lambda x: 1 <= x <= 100)
    fire_only = fields.Bool(missing=False)
    risk_level = fields.Str(missing=None, validate=lambda x: x in ['low', 'medium', 'high', 'critical'])
    start_date = fields.DateTime(missing=None)
    end_date = fields.DateTime(missing=None)

@detection_bp.route('/analyze', methods=['POST'])
@auth_service.optional_auth
def analyze_image():
    """Analyze uploaded image for fire detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    # Validate image
    is_valid, error_message = validate_image(file)
    if not is_valid:
        return jsonify({'error': error_message}), 400
    
    # Parse additional parameters
    try:
        form_data = request.form.to_dict()
        schema = DetectionRequestSchema()
        params = schema.load(form_data)
    except ValidationError as err:
        return jsonify({'error': 'Validation error', 'details': err.messages}), 400
    
    try:
        start_time = time.time()
        
        # Get current user
        current_user = getattr(request, 'current_user', None)
        user_id = current_user.id if current_user else None
        
        # Process and save image
        processed_image, image_path, filename = process_uploaded_image(file)
        if not processed_image:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Initialize AI service
        ai_service = AIService()
        
        # Perform fire detection
        detection_result = ai_service.detect_fire(processed_image)
        
        # Perform AI analysis if requested
        ai_analysis_result = None
        if params['use_ai_analysis'] and detection_result['fire_detected']:
            ai_analysis_result = ai_service.analyze_image_with_ai(
                processed_image,
                user_role=params['user_role'],
                user_location=params['user_location']
            )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Prepare detection record data
        detection_data = {
            'user_id': user_id,
            'image_filename': filename,
            'image_path': image_path,
            'fire_detected': detection_result['fire_detected'],
            'confidence_score': detection_result['confidence'],
            'detection_method': 'cnn' if not ai_analysis_result else 'combined',
            'user_role': params['user_role'],
            'user_location': params['user_location'],
            'processing_time': processing_time / 1000,  # Store in seconds
            'model_version': detection_result.get('model_version', 'unknown'),
            'session_id': request.headers.get('X-Session-ID'),
            'ip_address': request.remote_addr
        }
        
        # Add AI analysis results if available
        if ai_analysis_result:
            detection_data.update({
                'ai_analysis': ai_analysis_result.get('analysis'),
                'risk_level': ai_analysis_result.get('risk_level'),
                'fire_type': ai_analysis_result.get('fire_type'),
                'location_description': ai_analysis_result.get('location_description'),
                'recommendations': ai_analysis_result.get('recommendations'),
                'safety_instructions': ai_analysis_result.get('safety_instructions'),
                'ai_provider': ai_analysis_result.get('ai_provider', 'unknown')
            })
        
        # Save detection to database
        detection_repo = DetectionRepository()
        detection_record = detection_repo.create_detection(**detection_data)
        
        # Prepare response
        response_data = {
            'detection_id': detection_record.id,
            'fire_detected': detection_result['fire_detected'],
            'confidence': detection_result['confidence'],
            'processing_time_ms': processing_time,
            'timestamp': detection_record.created_at.isoformat()
        }
        
        # Add AI analysis to response if available
        if ai_analysis_result:
            response_data['ai_analysis'] = {
                'risk_level': ai_analysis_result.get('risk_level'),
                'fire_type': ai_analysis_result.get('fire_type'),
                'location_description': ai_analysis_result.get('location_description'),
                'recommendations': ai_analysis_result.get('recommendations'),
                'safety_instructions': ai_analysis_result.get('safety_instructions'),
                'full_analysis': ai_analysis_result.get('analysis')
            }
        
        status_code = 200 if not detection_result['fire_detected'] else 206  # 206 for fire detected
        return jsonify(response_data), status_code
        
    except Exception as e:
        current_app.logger.error(f"Detection analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@detection_bp.route('/history', methods=['GET'])
@jwt_required()
def get_detection_history():
    """Get detection history for current user"""
    current_user_id = get_jwt_identity()
    
    try:
        # Parse query parameters
        schema = DetectionHistorySchema()
        params = schema.load(request.args)
    except ValidationError as err:
        return jsonify({'error': 'Validation error', 'details': err.messages}), 400
    
    try:
        detection_repo = DetectionRepository()
        
        # Calculate offset
        offset = (params['page'] - 1) * params['per_page']
        
        # Get user's detection history
        detections = detection_repo.get_user_detections(
            user_id=current_user_id,
            limit=params['per_page'],
            offset=offset
        )
        
        # Apply filters if provided
        if params['fire_only']:
            detections = [d for d in detections if d.fire_detected]
        
        if params['risk_level']:
            detections = [d for d in detections if d.risk_level == params['risk_level']]
        
        if params['start_date'] or params['end_date']:
            filtered_detections = []
            for detection in detections:
                if params['start_date'] and detection.created_at < params['start_date']:
                    continue
                if params['end_date'] and detection.created_at > params['end_date']:
                    continue
                filtered_detections.append(detection)
            detections = filtered_detections
        
        # Convert to response format
        detection_list = [detection.to_dict(include_full_analysis=False) for detection in detections]
        
        # Get total count for pagination
        total_count = detection_repo.count(user_id=current_user_id)
        
        return jsonify({
            'detections': detection_list,
            'pagination': {
                'page': params['page'],
                'per_page': params['per_page'],
                'total': total_count,
                'pages': (total_count + params['per_page'] - 1) // params['per_page']
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"History retrieval error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve history'}), 500

@detection_bp.route('/history/<int:detection_id>', methods=['GET'])
@jwt_required()
def get_detection_detail(detection_id):
    """Get detailed information about a specific detection"""
    current_user_id = get_jwt_identity()
    
    try:
        detection_repo = DetectionRepository()
        detection = detection_repo.get_by_id(detection_id)
        
        if not detection:
            return jsonify({'error': 'Detection not found'}), 404
        
        # Check if user owns this detection or is admin
        user_repo = UserRepository()
        current_user = user_repo.get_by_id(current_user_id)
        
        if detection.user_id != current_user_id and not current_user.is_admin():
            return jsonify({'error': 'Access denied'}), 403
        
        return jsonify({
            'detection': detection.to_dict(include_full_analysis=True)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Detection detail error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve detection details'}), 500

@detection_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_detection_statistics():
    """Get detection statistics for current user"""
    current_user_id = get_jwt_identity()
    
    try:
        days = request.args.get('days', 30, type=int)
        if days < 1 or days > 365:
            return jsonify({'error': 'Days must be between 1 and 365'}), 400
        
        detection_repo = DetectionRepository()
        stats = detection_repo.get_detection_statistics(days=days, user_id=current_user_id)
        
        # Get daily counts
        daily_counts = detection_repo.get_daily_detection_counts(days=days)
        
        return jsonify({
            'statistics': stats,
            'daily_counts': daily_counts
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Statistics error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@detection_bp.route('/recent', methods=['GET'])
@auth_service.require_firefighter
def get_recent_detections():
    """Get recent fire detections (firefighter/admin only)"""
    try:
        hours = request.args.get('hours', 24, type=int)
        fire_only = request.args.get('fire_only', 'true').lower() == 'true'
        limit = request.args.get('limit', 50, type=int)
        
        if hours < 1 or hours > 168:  # Max 1 week
            return jsonify({'error': 'Hours must be between 1 and 168'}), 400
        
        if limit < 1 or limit > 100:
            return jsonify({'error': 'Limit must be between 1 and 100'}), 400
        
        detection_repo = DetectionRepository()
        detections = detection_repo.get_recent_detections(
            hours=hours,
            fire_only=fire_only,
            limit=limit
        )
        
        detection_list = [detection.to_dict(include_full_analysis=False) for detection in detections]
        
        return jsonify({
            'detections': detection_list,
            'count': len(detection_list),
            'filters': {
                'hours': hours,
                'fire_only': fire_only,
                'limit': limit
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Recent detections error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve recent detections'}), 500