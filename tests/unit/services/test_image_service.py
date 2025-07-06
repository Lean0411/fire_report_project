"""
測試 ImageService
"""
import pytest
from unittest.mock import patch, MagicMock, Mock, mock_open
import os
from PIL import Image
from datetime import datetime
from services.image_service import ImageService


class TestImageService:
    """ImageService 測試類"""
    
    @pytest.fixture
    def image_service(self):
        """創建 ImageService 實例"""
        with patch('services.image_service.Config') as mock_config:
            mock_config.UPLOAD_FOLDER = '/tmp/test_uploads'
            return ImageService()
    
    @pytest.fixture
    def mock_image(self):
        """創建模擬圖片"""
        img = MagicMock(spec=Image.Image)
        img.width = 800
        img.height = 600
        img.mode = 'RGB'
        img.format = 'JPEG'
        return img
    
    @patch('services.image_service.Image.open')
    @patch('services.image_service.ImageDraw.Draw')
    @patch('services.image_service.ImageFont.load_default')
    @patch('services.image_service.os.path.exists', return_value=True)
    def test_generate_annotated_image_fire_detected(self, mock_exists, mock_font, 
                                                   mock_draw, mock_open, 
                                                   image_service, mock_image):
        """測試生成火災檢測標註圖片"""
        # 設置模擬
        mock_open.return_value = mock_image
        mock_draw_obj = MagicMock()
        mock_draw.return_value = mock_draw_obj
        
        # 調用方法
        result = image_service.generate_annotated_image(
            '/tmp/test.jpg',
            'test.jpg',
            is_fire=True,
            p_fire=0.95,
            p_no=0.05
        )
        
        # 驗證
        assert result.endswith('_annotated.jpg')
        mock_open.assert_called_once()
        mock_draw_obj.rectangle.assert_called()
        mock_draw_obj.text.assert_called()
        mock_image.save.assert_called_once()
    
    @patch('services.image_service.Image.open')
    def test_generate_annotated_image_error(self, mock_open, image_service):
        """測試生成標註圖片時發生錯誤"""
        mock_open.side_effect = Exception('Image error')
        
        result = image_service.generate_annotated_image(
            '/tmp/test.jpg',
            'test.jpg',
            is_fire=False,
            p_fire=0.1,
            p_no=0.9
        )
        
        assert result is None
    
    @patch('services.image_service.Image.open')
    def test_resize_image_success(self, mock_open, image_service, mock_image):
        """測試成功調整圖片大小"""
        # 設置大圖片
        mock_image.width = 2000
        mock_image.height = 1500
        mock_open.return_value = mock_image
        
        result = image_service.resize_image('/tmp/large.jpg')
        
        # 驗證調整大小被調用
        mock_image.resize.assert_called_once()
        mock_image.save.assert_called_once()
        assert result == '/tmp/large.jpg'
    
    @patch('services.image_service.Image.open')
    def test_resize_image_small_image(self, mock_open, image_service, mock_image):
        """測試小圖片不需要調整大小"""
        # 設置小圖片
        mock_image.width = 800
        mock_image.height = 600
        mock_open.return_value = mock_image
        
        result = image_service.resize_image('/tmp/small.jpg')
        
        # 驗證沒有調整大小
        mock_image.resize.assert_not_called()
        assert result == '/tmp/small.jpg'
    
    @patch('services.image_service.Image.open')
    @patch('services.image_service.os.path.getsize')
    def test_get_image_info_success(self, mock_getsize, mock_open, 
                                   image_service, mock_image):
        """測試成功獲取圖片資訊"""
        mock_open.return_value = mock_image
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        result = image_service.get_image_info('/tmp/test.jpg')
        
        assert result is not None
        assert result['width'] == 800
        assert result['height'] == 600
        assert result['mode'] == 'RGB'
        assert result['format'] == 'JPEG'
        assert result['file_size'] == 1024 * 1024
        assert result['file_size_mb'] == 1.0
    
    @patch('services.image_service.Image.open')
    def test_get_image_info_error(self, mock_open, image_service):
        """測試獲取圖片資訊時發生錯誤"""
        mock_open.side_effect = Exception('Read error')
        
        result = image_service.get_image_info('/tmp/test.jpg')
        
        assert result is None
    
    @patch('services.image_service.cleanup_old_files')
    def test_cleanup_old_images(self, mock_cleanup, image_service):
        """測試清理舊圖片"""
        mock_cleanup.return_value = 5
        
        result = image_service.cleanup_old_images()
        
        assert result == 5
        mock_cleanup.assert_called_once_with(
            '/tmp/test_uploads',
            max_age_hours=24,
            max_files=100
        )
    
    @patch('services.image_service.Image.open')
    def test_convert_to_web_format_png(self, mock_open, image_service):
        """測試轉換 PNG 到網頁格式"""
        # 創建 PNG 圖片
        mock_img = MagicMock()
        mock_img.format = 'PNG'
        mock_open.return_value = mock_img
        
        result = image_service.convert_to_web_format('/tmp/test.png')
        
        # 驗證保存為 JPEG
        assert result.endswith('.jpg')
        mock_img.save.assert_called_once()
        save_args = mock_img.save.call_args
        assert save_args[0][1] == 'JPEG'
    
    @patch('services.image_service.Image.open')
    def test_convert_to_web_format_already_jpeg(self, mock_open, image_service):
        """測試 JPEG 圖片不需要轉換"""
        # 創建 JPEG 圖片
        mock_img = MagicMock()
        mock_img.format = 'JPEG'
        mock_open.return_value = mock_img
        
        result = image_service.convert_to_web_format('/tmp/test.jpg')
        
        # 驗證返回原路徑
        assert result == '/tmp/test.jpg'
        mock_img.save.assert_not_called()
    
    @patch('services.image_service.Image.open')
    def test_convert_to_web_format_error(self, mock_open, image_service):
        """測試轉換格式時發生錯誤"""
        mock_open.side_effect = IOError('File error')
        
        result = image_service.convert_to_web_format('/tmp/test.png')
        
        # 驗證返回原路徑
        assert result == '/tmp/test.png'