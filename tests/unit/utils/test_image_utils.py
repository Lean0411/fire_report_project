"""
測試 image_utils 模組
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import os
import numpy as np
from PIL import Image
from io import BytesIO
from werkzeug.datastructures import FileStorage
from utils.image_utils import (
    allowed_file, get_file_extension, validate_image,
    generate_unique_filename, ensure_upload_folder,
    process_uploaded_image, delete_image, get_image_info,
    create_thumbnail
)


class TestImageUtils:
    """ImageUtils 測試類"""
    
    def test_allowed_file_valid_extensions(self):
        """測試允許的檔案副檔名"""
        assert allowed_file('image.jpg') is True
        assert allowed_file('photo.jpeg') is True
        assert allowed_file('picture.png') is True
        assert allowed_file('IMAGE.JPG') is True
    
    def test_allowed_file_invalid_extensions(self):
        """測試不允許的檔案副檔名"""
        assert allowed_file('document.pdf') is False
        assert allowed_file('script.exe') is False
        assert allowed_file('no_extension') is False
    
    def test_get_file_extension(self):
        """測試獲取檔案副檔名"""
        assert get_file_extension('image.jpg') == 'jpg'
        assert get_file_extension('photo.JPEG') == 'jpeg'
        assert get_file_extension('no_extension') == ''
        assert get_file_extension('multiple.dots.png') == 'png'
    
    def create_mock_file(self, filename='test.jpg', size=1024, content=None):
        """創建模擬文件對象"""
        if content is None:
            # 創建一個真實的圖片內容
            img = Image.new('RGB', (100, 100), color='red')
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            content = img_bytes.getvalue()
            
        file = Mock(spec=FileStorage)
        file.filename = filename
        file.tell = Mock(return_value=size)
        file.seek = Mock()
        file.read = Mock(return_value=content)
        file.save = Mock()
        
        # 模擬 file-like 行為
        file_stream = BytesIO(content)
        file.__enter__ = Mock(return_value=file_stream)
        file.__exit__ = Mock(return_value=None)
        
        # 讓 Image.open 能夠工作
        def seek_side_effect(pos, whence=0):
            file_stream.seek(pos, whence)
        file.seek.side_effect = seek_side_effect
        
        # 模擬 context manager
        file.stream = file_stream
        return file, file_stream
    
    @patch('utils.image_utils.Image.open')
    def test_validate_image_valid(self, mock_image_open):
        """測試驗證有效圖片"""
        # 創建模擬圖片
        mock_image = Mock()
        mock_image.size = (800, 600)
        mock_image.verify = Mock()
        mock_image_open.return_value = mock_image
        
        file, _ = self.create_mock_file('test.jpg', 1024 * 1024)  # 1MB
        
        is_valid, error = validate_image(file)
        assert is_valid is True
        assert error is None
    
    def test_validate_image_no_filename(self):
        """測試驗證無文件名"""
        file = Mock(spec=FileStorage)
        file.filename = ''
        
        is_valid, error = validate_image(file)
        assert is_valid is False
        assert error == "文件名為空"
    
    def test_validate_image_invalid_extension(self):
        """測試驗證無效擴展名"""
        file = Mock(spec=FileStorage)
        file.filename = 'test.exe'
        
        is_valid, error = validate_image(file)
        assert is_valid is False
        assert "不支持的文件格式" in error
    
    def test_validate_image_too_large(self):
        """測試驗證文件太大"""
        file, _ = self.create_mock_file('test.jpg', 10 * 1024 * 1024)  # 10MB
        
        is_valid, error = validate_image(file)
        assert is_valid is False
        assert "文件大小超過限制" in error
    
    def test_validate_image_empty_file(self):
        """測試驗證空文件"""
        file, _ = self.create_mock_file('test.jpg', 0)
        
        is_valid, error = validate_image(file)
        assert is_valid is False
        assert error == "文件為空"
    
    @patch('utils.image_utils.Image.open')
    def test_validate_image_too_small(self, mock_image_open):
        """測試驗證圖片太小"""
        mock_image = Mock()
        mock_image.size = (5, 5)
        mock_image.verify = Mock()
        mock_image_open.return_value = mock_image
        
        file, _ = self.create_mock_file('test.jpg')
        
        is_valid, error = validate_image(file)
        assert is_valid is False
        assert "圖片尺寸太小" in error
    
    @patch('utils.image_utils.Image.open')
    def test_validate_image_too_large_dimensions(self, mock_image_open):
        """測試驗證圖片尺寸太大"""
        mock_image = Mock()
        mock_image.size = (15000, 15000)
        mock_image.verify = Mock()
        mock_image_open.return_value = mock_image
        
        file, _ = self.create_mock_file('test.jpg')
        
        is_valid, error = validate_image(file)
        assert is_valid is False
        assert "圖片尺寸太大" in error
    
    @patch('utils.image_utils.Image.open')
    def test_validate_image_invalid_format(self, mock_image_open):
        """測試驗證無效圖片格式"""
        mock_image_open.side_effect = Exception("Invalid image")
        
        file, _ = self.create_mock_file('test.jpg')
        
        is_valid, error = validate_image(file)
        assert is_valid is False
        assert error == "無效的圖片文件"
    
    def test_generate_unique_filename(self):
        """測試生成唯一文件名"""
        filename1 = generate_unique_filename('test.jpg')
        filename2 = generate_unique_filename('test.jpg')
        
        assert filename1 != filename2
        assert filename1.endswith('.jpg')
        assert filename2.endswith('.jpg')
        assert len(filename1) > len('test.jpg')
    
    def test_generate_unique_filename_no_extension(self):
        """測試生成唯一文件名（無擴展名）"""
        filename = generate_unique_filename('testfile')
        assert '.' not in filename
        assert len(filename) > len('testfile')
    
    @patch('utils.image_utils.os.makedirs')
    @patch('utils.image_utils.os.path.exists')
    def test_ensure_upload_folder_creates(self, mock_exists, mock_makedirs):
        """測試確保上傳文件夾存在 - 創建新文件夾"""
        mock_exists.return_value = False
        
        result = ensure_upload_folder()
        
        assert result is True
        mock_makedirs.assert_called_once()
    
    @patch('utils.image_utils.os.path.exists')
    def test_ensure_upload_folder_exists(self, mock_exists):
        """測試確保上傳文件夾存在 - 文件夾已存在"""
        mock_exists.return_value = True
        
        result = ensure_upload_folder()
        
        assert result is True
    
    @patch('utils.image_utils.os.makedirs')
    @patch('utils.image_utils.os.path.exists')
    def test_ensure_upload_folder_error(self, mock_exists, mock_makedirs):
        """測試確保上傳文件夾存在 - 創建失敗"""
        mock_exists.return_value = False
        mock_makedirs.side_effect = OSError("Permission denied")
        
        result = ensure_upload_folder()
        
        assert result is False
    
    @patch('utils.image_utils.ensure_upload_folder')
    @patch('utils.image_utils.Image.open')
    @patch('utils.image_utils.np.array')
    def test_process_uploaded_image_success(self, mock_np_array, mock_image_open, mock_ensure):
        """測試處理上傳圖片 - 成功"""
        mock_ensure.return_value = True
        
        # 創建模擬圖片
        mock_image = Mock()
        mock_image.mode = 'RGB'
        mock_image.size = (800, 600)
        mock_image.save = Mock()
        mock_image_open.return_value = mock_image
        
        # 創建模擬數組
        mock_array = np.zeros((600, 800, 3))
        mock_np_array.return_value = mock_array
        
        file, _ = self.create_mock_file('test.jpg')
        
        image_array, path, filename = process_uploaded_image(file)
        
        assert image_array is not None
        assert path is not None
        assert filename is not None
        assert filename.endswith('.jpg')
        file.save.assert_called_once()
    
    @patch('utils.image_utils.ensure_upload_folder')
    def test_process_uploaded_image_no_folder(self, mock_ensure):
        """測試處理上傳圖片 - 無法創建文件夾"""
        mock_ensure.return_value = False
        
        file, _ = self.create_mock_file('test.jpg')
        
        image_array, path, filename = process_uploaded_image(file)
        
        assert image_array is None
        assert path is None
        assert filename is None
    
    @patch('utils.image_utils.ensure_upload_folder')
    @patch('utils.image_utils.Image.open')
    def test_process_uploaded_image_convert_mode(self, mock_image_open, mock_ensure):
        """測試處理上傳圖片 - 轉換模式"""
        mock_ensure.return_value = True
        
        # 創建 RGBA 模式的圖片
        mock_image = Mock()
        mock_image.mode = 'RGBA'
        mock_image.size = (800, 600)
        mock_image.save = Mock()
        mock_image.convert = Mock(return_value=mock_image)
        mock_image_open.return_value = mock_image
        
        file, _ = self.create_mock_file('test.png')
        
        process_uploaded_image(file)
        
        mock_image.convert.assert_called_once_with('RGB')
    
    @patch('utils.image_utils.ensure_upload_folder')
    @patch('utils.image_utils.Image.open')
    def test_process_uploaded_image_resize(self, mock_image_open, mock_ensure):
        """測試處理上傳圖片 - 調整大小"""
        mock_ensure.return_value = True
        
        # 創建大圖片
        mock_image = Mock()
        mock_image.mode = 'RGB'
        mock_image.size = (2000, 1500)
        mock_image.save = Mock()
        mock_image.resize = Mock(return_value=mock_image)
        mock_image_open.return_value = mock_image
        
        file, _ = self.create_mock_file('test.jpg')
        
        process_uploaded_image(file)
        
        mock_image.resize.assert_called_once()
    
    @patch('utils.image_utils.ensure_upload_folder')
    @patch('utils.image_utils.os.remove')
    def test_process_uploaded_image_error_cleanup(self, mock_remove, mock_ensure):
        """測試處理上傳圖片 - 錯誤清理"""
        mock_ensure.return_value = True
        
        file, _ = self.create_mock_file('test.jpg')
        file.save.side_effect = Exception("Save failed")
        
        image_array, path, filename = process_uploaded_image(file)
        
        assert image_array is None
        assert path is None
        assert filename is None
    
    @patch('utils.image_utils.os.remove')
    @patch('utils.image_utils.os.path.exists')
    def test_delete_image_success(self, mock_exists, mock_remove):
        """測試刪除圖片 - 成功"""
        mock_exists.return_value = True
        
        result = delete_image('/path/to/image.jpg')
        
        assert result is True
        mock_remove.assert_called_once_with('/path/to/image.jpg')
    
    @patch('utils.image_utils.os.path.exists')
    def test_delete_image_not_exists(self, mock_exists):
        """測試刪除圖片 - 文件不存在"""
        mock_exists.return_value = False
        
        result = delete_image('/path/to/image.jpg')
        
        assert result is False
    
    @patch('utils.image_utils.os.remove')
    @patch('utils.image_utils.os.path.exists')
    def test_delete_image_error(self, mock_exists, mock_remove):
        """測試刪除圖片 - 刪除失敗"""
        mock_exists.return_value = True
        mock_remove.side_effect = OSError("Permission denied")
        
        result = delete_image('/path/to/image.jpg')
        
        assert result is False
    
    @patch('utils.image_utils.Image.open')
    @patch('utils.image_utils.os.path.getsize')
    @patch('utils.image_utils.os.path.exists')
    def test_get_image_info_success(self, mock_exists, mock_getsize, mock_image_open):
        """測試獲取圖片信息 - 成功"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        mock_image = Mock()
        mock_image.width = 800
        mock_image.height = 600
        mock_image.format = 'JPEG'
        mock_image.mode = 'RGB'
        mock_image_open.return_value = mock_image
        
        info = get_image_info('/path/to/image.jpg')
        
        assert info is not None
        assert info['width'] == 800
        assert info['height'] == 600
        assert info['format'] == 'JPEG'
        assert info['mode'] == 'RGB'
        assert info['size_bytes'] == 1024 * 1024
        assert info['size_mb'] == 1.0
    
    @patch('utils.image_utils.os.path.exists')
    def test_get_image_info_not_exists(self, mock_exists):
        """測試獲取圖片信息 - 文件不存在"""
        mock_exists.return_value = False
        
        info = get_image_info('/path/to/image.jpg')
        
        assert info is None
    
    @patch('utils.image_utils.Image.open')
    @patch('utils.image_utils.os.path.exists')
    def test_get_image_info_error(self, mock_exists, mock_image_open):
        """測試獲取圖片信息 - 錯誤"""
        mock_exists.return_value = True
        mock_image_open.side_effect = Exception("Cannot open image")
        
        info = get_image_info('/path/to/image.jpg')
        
        assert info is None
    
    @patch('utils.image_utils.Image.open')
    @patch('utils.image_utils.os.path.exists')
    def test_create_thumbnail_success(self, mock_exists, mock_image_open):
        """測試創建縮略圖 - 成功"""
        mock_exists.return_value = True
        
        mock_image = Mock()
        mock_image.thumbnail = Mock()
        mock_image.save = Mock()
        mock_image_open.return_value = mock_image
        
        thumbnail_path = create_thumbnail('/path/to/image.jpg')
        
        assert thumbnail_path is not None
        assert '_thumb.jpg' in thumbnail_path
        mock_image.thumbnail.assert_called_once()
        mock_image.save.assert_called_once()
    
    @patch('utils.image_utils.os.path.exists')
    def test_create_thumbnail_not_exists(self, mock_exists):
        """測試創建縮略圖 - 文件不存在"""
        mock_exists.return_value = False
        
        thumbnail_path = create_thumbnail('/path/to/image.jpg')
        
        assert thumbnail_path is None
    
    @patch('utils.image_utils.Image.open')
    @patch('utils.image_utils.os.path.exists')
    def test_create_thumbnail_error(self, mock_exists, mock_image_open):
        """測試創建縮略圖 - 錯誤"""
        mock_exists.return_value = True
        mock_image_open.side_effect = Exception("Cannot create thumbnail")
        
        thumbnail_path = create_thumbnail('/path/to/image.jpg')
        
        assert thumbnail_path is None
    
    @patch('utils.image_utils.Image.open')
    @patch('utils.image_utils.os.path.exists')
    def test_create_thumbnail_custom_size(self, mock_exists, mock_image_open):
        """測試創建縮略圖 - 自定義大小"""
        mock_exists.return_value = True
        
        mock_image = Mock()
        mock_image.thumbnail = Mock()
        mock_image.save = Mock()
        mock_image_open.return_value = mock_image
        
        thumbnail_path = create_thumbnail('/path/to/image.jpg', size=(256, 256))
        
        assert thumbnail_path is not None
        mock_image.thumbnail.assert_called_once_with((256, 256), Image.Resampling.LANCZOS)