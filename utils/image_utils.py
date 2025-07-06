"""
圖片處理工具模組
處理圖片驗證、轉換、保存等功能
"""
import os
import time
import hashlib
from typing import Tuple, Optional, Any
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from config.settings import ALLOWED_EXTENSIONS, UPLOAD_FOLDER, MAX_FILE_SIZE
import logging

logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """檢查文件是否為允許的格式"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """獲取文件擴展名"""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''


def validate_image(file: FileStorage) -> Tuple[bool, Optional[str]]:
    """
    驗證上傳的圖片文件
    
    Args:
        file: 上傳的文件對象
        
    Returns:
        Tuple[bool, Optional[str]]: (是否有效, 錯誤信息)
    """
    # 檢查文件名
    if not file.filename:
        return False, "文件名為空"
    
    # 檢查文件擴展名
    if not allowed_file(file.filename):
        return False, f"不支持的文件格式。支持的格式：{', '.join(ALLOWED_EXTENSIONS)}"
    
    # 檢查文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
        return False, f"文件大小超過限制（最大 {max_size_mb:.1f} MB）"
    
    if file_size == 0:
        return False, "文件為空"
    
    # 嘗試打開圖片以驗證格式
    try:
        image = Image.open(file)
        image.verify()  # 驗證圖片完整性
        file.seek(0)  # Reset file pointer
        
        # 檢查圖片尺寸
        image = Image.open(file)
        width, height = image.size
        if width < 10 or height < 10:
            return False, "圖片尺寸太小（最小 10x10 像素）"
        
        if width > 10000 or height > 10000:
            return False, "圖片尺寸太大（最大 10000x10000 像素）"
        
        file.seek(0)  # Reset file pointer again
        
    except Exception as e:
        logger.error(f"圖片驗證失敗: {str(e)}")
        return False, "無效的圖片文件"
    
    return True, None


def generate_unique_filename(original_filename: str) -> str:
    """
    生成唯一的文件名
    
    Args:
        original_filename: 原始文件名
        
    Returns:
        str: 唯一的文件名
    """
    timestamp = str(int(time.time() * 1000))
    random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    extension = get_file_extension(original_filename)
    
    if extension:
        return f"{timestamp}_{random_str}.{extension}"
    else:
        return f"{timestamp}_{random_str}"


def ensure_upload_folder() -> bool:
    """確保上傳文件夾存在"""
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"創建上傳文件夾失敗: {str(e)}")
        return False


def process_uploaded_image(file: FileStorage) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    處理上傳的圖片文件
    
    Args:
        file: 上傳的文件對象
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[str], Optional[str]]: 
        (處理後的圖片數組, 圖片保存路徑, 文件名)
    """
    try:
        # 確保上傳文件夾存在
        if not ensure_upload_folder():
            logger.error("無法創建上傳文件夾")
            return None, None, None
        
        # 生成安全的文件名
        original_filename = secure_filename(file.filename)
        filename = generate_unique_filename(original_filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存文件
        file.save(filepath)
        logger.info(f"圖片已保存: {filepath}")
        
        # 讀取並處理圖片
        image = Image.open(filepath)
        
        # 轉換為 RGB 格式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 調整圖片大小以優化處理速度
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 保存調整大小後的圖片
            image.save(filepath, quality=95)
            logger.info(f"圖片已調整大小: {image.size}")
        
        # 轉換為 numpy 數組
        image_array = np.array(image)
        
        # 相對路徑（用於數據庫存儲）
        relative_path = os.path.relpath(filepath, os.path.dirname(UPLOAD_FOLDER))
        
        return image_array, relative_path, filename
        
    except Exception as e:
        logger.error(f"處理圖片失敗: {str(e)}")
        # 嘗試清理已保存的文件
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return None, None, None


def delete_image(filepath: str) -> bool:
    """
    刪除圖片文件
    
    Args:
        filepath: 文件路徑
        
    Returns:
        bool: 是否成功刪除
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"圖片已刪除: {filepath}")
            return True
        else:
            logger.warning(f"圖片不存在: {filepath}")
            return False
    except Exception as e:
        logger.error(f"刪除圖片失敗: {str(e)}")
        return False


def get_image_info(filepath: str) -> Optional[dict]:
    """
    獲取圖片信息
    
    Args:
        filepath: 文件路徑
        
    Returns:
        Optional[dict]: 圖片信息字典
    """
    try:
        if not os.path.exists(filepath):
            return None
        
        image = Image.open(filepath)
        file_size = os.path.getsize(filepath)
        
        return {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        logger.error(f"獲取圖片信息失敗: {str(e)}")
        return None


def create_thumbnail(filepath: str, size: Tuple[int, int] = (128, 128)) -> Optional[str]:
    """
    創建縮略圖
    
    Args:
        filepath: 原始圖片路徑
        size: 縮略圖大小
        
    Returns:
        Optional[str]: 縮略圖路徑
    """
    try:
        if not os.path.exists(filepath):
            return None
        
        image = Image.open(filepath)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # 生成縮略圖文件名
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        thumbnail_filename = f"{name}_thumb{ext}"
        thumbnail_path = os.path.join(directory, thumbnail_filename)
        
        # 保存縮略圖
        image.save(thumbnail_path, quality=85)
        logger.info(f"縮略圖已創建: {thumbnail_path}")
        
        return thumbnail_path
        
    except Exception as e:
        logger.error(f"創建縮略圖失敗: {str(e)}")
        return None