"""
檔案處理工具模組
提供檔案驗證、上傳和處理相關功能
"""
import os
from werkzeug.utils import secure_filename
from .constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE

def allowed_file(filename: str) -> bool:
    """
    檢查檔案是否為允許的格式
    
    Args:
        filename: 檔案名稱
        
    Returns:
        bool: 是否為允許的檔案格式
    """
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def validate_file(file) -> tuple[bool, str]:
    """
    驗證上傳檔案的格式和大小
    
    Args:
        file: 上傳的檔案物件
        
    Returns:
        tuple: (是否有效, 錯誤訊息)
    """
    if not file:
        return False, "未選擇檔案"
    
    if file.filename == '':
        return False, "檔案名稱為空"
    
    if not allowed_file(file.filename):
        return False, f"不支援的檔案格式，請上傳 {', '.join(ALLOWED_EXTENSIONS)} 格式"
    
    # 檢查檔案大小（需要先讀取）
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # 重置檔案指標
    
    if file_size > MAX_FILE_SIZE:
        size_mb = MAX_FILE_SIZE / (1024 * 1024)
        return False, f"檔案大小超過限制 ({size_mb}MB)"
    
    return True, ""

def secure_save_file(file, upload_folder: str) -> tuple[bool, str, str]:
    """
    安全地儲存上傳檔案
    
    Args:
        file: 上傳的檔案物件
        upload_folder: 上傳目錄
        
    Returns:
        tuple: (是否成功, 檔案路徑, 錯誤訊息)
    """
    try:
        # 驗證檔案
        is_valid, error_msg = validate_file(file)
        if not is_valid:
            return False, "", error_msg
        
        # 生成安全的檔案名
        filename = secure_filename(file.filename)
        
        # 確保上傳目錄存在
        os.makedirs(upload_folder, exist_ok=True)
        
        # 生成唯一檔案名避免衝突
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        unique_filename = f"{timestamp}.{ext.lstrip('.')}"
        
        filepath = os.path.join(upload_folder, unique_filename)
        
        # 儲存檔案
        file.save(filepath)
        
        return True, filepath, ""
        
    except Exception as e:
        return False, "", f"檔案儲存失敗: {str(e)}"

def cleanup_old_files(directory: str, max_age_hours: int = 24, max_files: int = 100):
    """
    清理舊檔案
    
    Args:
        directory: 要清理的目錄
        max_age_hours: 檔案最大年齡（小時）
        max_files: 最大檔案數量
    """
    try:
        import time
        
        if not os.path.exists(directory):
            return
        
        files = []
        current_time = time.time()
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getctime(filepath)
                files.append((filepath, file_age))
        
        # 按年齡排序
        files.sort(key=lambda x: x[1], reverse=True)
        
        # 刪除過舊的檔案
        max_age_seconds = max_age_hours * 3600
        for filepath, age in files:
            if age > max_age_seconds:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
        
        # 如果檔案太多，刪除最舊的
        remaining_files = [f for f, a in files if a <= max_age_seconds]
        if len(remaining_files) > max_files:
            files_to_remove = remaining_files[max_files:]
            for filepath in files_to_remove:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                    
    except Exception:
        # 清理失敗不影響主要功能
        pass