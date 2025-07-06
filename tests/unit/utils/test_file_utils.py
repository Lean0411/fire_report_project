"""
測試 file_utils 模組
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
from datetime import datetime, timedelta
from werkzeug.datastructures import FileStorage
from utils.file_utils import (
    allowed_file, secure_save_file, get_file_extension,
    validate_file_size, ensure_upload_directory,
    cleanup_old_files, generate_unique_filename
)


class TestFileUtils:
    """FileUtils 測試類"""
    
    def test_allowed_file_valid_extensions(self):
        """測試允許的檔案副檔名"""
        assert allowed_file('image.jpg') is True
        assert allowed_file('photo.jpeg') is True
        assert allowed_file('picture.png') is True
        assert allowed_file('IMAGE.JPG') is True  # 測試大寫
    
    def test_allowed_file_invalid_extensions(self):
        """測試不允許的檔案副檔名"""
        assert allowed_file('document.pdf') is False
        assert allowed_file('script.exe') is False
        assert allowed_file('data.txt') is False
        assert allowed_file('no_extension') is False
    
    def test_allowed_file_edge_cases(self):
        """測試邊界情況"""
        assert allowed_file('') is False
        assert allowed_file('.jpg') is True  # 隱藏檔案
        assert allowed_file('file.name.jpg') is True  # 多個點
        assert allowed_file('file.JPG.exe') is False  # 偽裝的執行檔
    
    def test_get_file_extension(self):
        """測試獲取檔案副檔名"""
        assert get_file_extension('image.jpg') == 'jpg'
        assert get_file_extension('photo.JPEG') == 'jpeg'
        assert get_file_extension('document.pdf') == 'pdf'
        assert get_file_extension('no_extension') == ''
        assert get_file_extension('multiple.dots.png') == 'png'
    
    @patch('utils.file_utils.os.path.getsize')
    def test_validate_file_size_valid(self, mock_getsize):
        """測試檔案大小驗證 - 有效"""
        mock_getsize.return_value = 1024 * 1024  # 1MB
        assert validate_file_size('test.jpg') is True
    
    @patch('utils.file_utils.os.path.getsize')
    def test_validate_file_size_too_large(self, mock_getsize):
        """測試檔案大小驗證 - 太大"""
        mock_getsize.return_value = 10 * 1024 * 1024  # 10MB
        assert validate_file_size('test.jpg') is False
    
    @patch('utils.file_utils.os.path.getsize')
    def test_validate_file_size_error(self, mock_getsize):
        """測試檔案大小驗證 - 錯誤"""
        mock_getsize.side_effect = OSError('File not found')
        assert validate_file_size('nonexistent.jpg') is False
    
    @patch('utils.file_utils.os.makedirs')
    @patch('utils.file_utils.os.path.exists')
    def test_ensure_upload_directory_creates(self, mock_exists, mock_makedirs):
        """測試確保上傳目錄存在 - 創建新目錄"""
        mock_exists.return_value = False
        
        result = ensure_upload_directory('/tmp/uploads')
        
        assert result is True
        mock_makedirs.assert_called_once_with('/tmp/uploads', exist_ok=True)
    
    @patch('utils.file_utils.os.path.exists')
    def test_ensure_upload_directory_exists(self, mock_exists):
        """測試確保上傳目錄存在 - 目錄已存在"""
        mock_exists.return_value = True
        
        result = ensure_upload_directory('/tmp/uploads')
        
        assert result is True
    
    @patch('utils.file_utils.os.makedirs')
    @patch('utils.file_utils.os.path.exists')
    def test_ensure_upload_directory_error(self, mock_exists, mock_makedirs):
        """測試確保上傳目錄存在 - 創建失敗"""
        mock_exists.return_value = False
        mock_makedirs.side_effect = OSError('Permission denied')
        
        result = ensure_upload_directory('/tmp/uploads')
        
        assert result is False
    
    def test_generate_unique_filename(self):
        """測試生成唯一檔名"""
        result1 = generate_unique_filename('test.jpg')
        result2 = generate_unique_filename('test.jpg')
        
        # 確保生成的檔名不同
        assert result1 != result2
        
        # 確保保留副檔名
        assert result1.endswith('.jpg')
        assert result2.endswith('.jpg')
        
        # 確保包含時間戳
        assert len(result1) > len('test.jpg')
    
    def test_generate_unique_filename_no_extension(self):
        """測試生成唯一檔名 - 無副檔名"""
        result = generate_unique_filename('testfile')
        assert '.' not in result
        assert len(result) > len('testfile')
    
    @patch('utils.file_utils.ensure_upload_directory')
    @patch('utils.file_utils.validate_filename')
    @patch('utils.file_utils.generate_unique_filename')
    def test_secure_save_file_success(self, mock_unique, mock_validate, mock_ensure):
        """測試安全保存檔案 - 成功"""
        # 設置 mocks
        mock_ensure.return_value = True
        mock_validate.return_value = True
        mock_unique.return_value = 'unique_test.jpg'
        
        # 創建模擬檔案
        mock_file = MagicMock(spec=FileStorage)
        mock_file.filename = 'test.jpg'
        mock_file.save = MagicMock()
        
        # 調用函數
        result = secure_save_file(mock_file, '/tmp/uploads')
        
        # 驗證結果
        assert result == '/tmp/uploads/unique_test.jpg'
        mock_file.save.assert_called_once()
    
    @patch('utils.file_utils.ensure_upload_directory')
    def test_secure_save_file_no_directory(self, mock_ensure):
        """測試安全保存檔案 - 無法創建目錄"""
        mock_ensure.return_value = False
        mock_file = MagicMock(spec=FileStorage)
        
        result = secure_save_file(mock_file, '/tmp/uploads')
        
        assert result is None
    
    @patch('utils.file_utils.validate_filename')
    @patch('utils.file_utils.ensure_upload_directory')
    def test_secure_save_file_invalid_filename(self, mock_ensure, mock_validate):
        """測試安全保存檔案 - 無效檔名"""
        mock_ensure.return_value = True
        mock_validate.return_value = False
        
        mock_file = MagicMock(spec=FileStorage)
        mock_file.filename = '../../../etc/passwd'
        
        result = secure_save_file(mock_file, '/tmp/uploads')
        
        assert result is None
    
    @patch('utils.file_utils.os.walk')
    @patch('utils.file_utils.os.path.getmtime')
    @patch('utils.file_utils.os.remove')
    def test_cleanup_old_files_by_age(self, mock_remove, mock_getmtime, mock_walk):
        """測試清理舊檔案 - 按年齡"""
        # 設置測試數據
        now = datetime.now().timestamp()
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        
        mock_walk.return_value = [
            ('/tmp/uploads', [], ['old_file.jpg', 'new_file.jpg'])
        ]
        mock_getmtime.side_effect = [old_time, now]
        
        # 調用函數
        result = cleanup_old_files('/tmp/uploads', max_age_hours=24)
        
        # 驗證結果
        assert result == 1
        mock_remove.assert_called_once_with('/tmp/uploads/old_file.jpg')
    
    @patch('utils.file_utils.os.walk')
    @patch('utils.file_utils.os.path.getmtime')
    @patch('utils.file_utils.os.remove')
    def test_cleanup_old_files_by_count(self, mock_remove, mock_getmtime, mock_walk):
        """測試清理舊檔案 - 按數量"""
        # 設置測試數據
        now = datetime.now().timestamp()
        
        mock_walk.return_value = [
            ('/tmp/uploads', [], ['file1.jpg', 'file2.jpg', 'file3.jpg'])
        ]
        mock_getmtime.side_effect = [now - 300, now - 200, now - 100]  # 不同時間
        
        # 調用函數，最多保留2個檔案
        result = cleanup_old_files('/tmp/uploads', max_files=2)
        
        # 驗證結果 - 應該刪除最舊的檔案
        assert result == 1
        mock_remove.assert_called_once_with('/tmp/uploads/file1.jpg')
    
    @patch('utils.file_utils.os.walk')
    def test_cleanup_old_files_empty_directory(self, mock_walk):
        """測試清理舊檔案 - 空目錄"""
        mock_walk.return_value = [('/tmp/uploads', [], [])]
        
        result = cleanup_old_files('/tmp/uploads')
        
        assert result == 0
    
    @patch('utils.file_utils.os.walk')
    @patch('utils.file_utils.os.remove')
    def test_cleanup_old_files_remove_error(self, mock_remove, mock_walk):
        """測試清理舊檔案 - 刪除錯誤"""
        mock_walk.return_value = [
            ('/tmp/uploads', [], ['protected_file.jpg'])
        ]
        mock_remove.side_effect = OSError('Permission denied')
        
        # 應該繼續執行，不拋出異常
        result = cleanup_old_files('/tmp/uploads', max_age_hours=0)
        
        assert result == 0  # 刪除失敗，計數為0