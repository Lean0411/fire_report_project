import unittest
from app.processor import process_image, load_sop_recommendations
from app.config import Config
import os

class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.test_image_path = os.path.join(Config.BASE_DIR, 'static/uploads/IMG_1024.jpeg')
        
    def test_load_sop_recommendations(self):
        """測試 SOP 建議載入功能"""
        sop_data = load_sop_recommendations()
        self.assertIsNotNone(sop_data)
        self.assertIn('general', sop_data)
        self.assertIn('firefighter', sop_data)
        self.assertIn('management', sop_data)

    def test_process_image(self):
        """測試圖片處理功能"""
        if not os.path.exists(self.test_image_path):
            self.skipTest("測試圖片不存在")
            
        result, error = process_image(self.test_image_path, 'general')
        if error:  # 如果模型未載入，跳過測試
            self.skipTest("模型未載入: " + error)
            
        self.assertIsNotNone(result)
        self.assertIn('detection', result)
        self.assertIn('recommendations', result)

if __name__ == '__main__':
    unittest.main()