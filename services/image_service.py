"""
圖片服務模組
提供圖片處理、標註和管理相關功能
"""
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
from datetime import datetime

from config.settings import Config
from config.logging_config import get_logger
from config.constants import (
    IMAGE_QUALITY_HIGH, IMAGE_QUALITY_MEDIUM, IMAGE_MAX_SIZE,
    FONT_SIZE_MIN, FONT_SIZE_DIVISOR, UI_PADDING,
    COLOR_FIRE_RED, COLOR_SAFE_GREEN, COLOR_WHITE,
    ALPHA_SEMI_TRANSPARENT, IMAGE_CLEANUP_MAX_AGE_HOURS,
    IMAGE_CLEANUP_MAX_FILES, BYTES_TO_MB
)
from utils.file_utils import cleanup_old_files

logger = get_logger(__name__)

class ImageService:
    """圖片服務類，負責圖片處理和標註"""
    
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER
        
    def generate_annotated_image(self, image_path: str, filename: str, 
                               is_fire: bool, p_fire: float, p_no: float,
                               add_annotations: bool = False) -> str:
        """
        生成分析結果圖片
        
        Args:
            image_path: 原始圖片路徑
            filename: 檔案名稱
            is_fire: 是否檢測到火災
            p_fire: 火災機率
            p_no: 安全機率
            add_annotations: 是否添加標註
            
        Returns:
            str: 生成的圖片檔名
        """
        try:
            # 開啟原圖
            img = Image.open(image_path).convert('RGB')
            
            if add_annotations:
                img = self._add_annotations(img, is_fire, p_fire, p_no)
            
            # 生成新檔名
            annotated_filename = f"annotated_{filename}"
            annotated_path = os.path.join(self.upload_folder, annotated_filename)
            
            # 儲存圖片
            img.save(annotated_path, quality=IMAGE_QUALITY_HIGH)
            
            logger.info(f"成功生成分析圖片: {annotated_filename}")
            return annotated_filename
            
        except Exception as e:
            logger.error(f"生成分析圖片時發生錯誤: {e}")
            return filename  # 如果失敗，返回原檔名
    
    def _add_annotations(self, img: Image.Image, is_fire: bool, 
                        p_fire: float, p_no: float) -> Image.Image:
        """
        在圖片上添加檢測結果標註
        
        Args:
            img: PIL 圖片物件
            is_fire: 是否檢測到火災
            p_fire: 火災機率
            p_no: 安全機率
            
        Returns:
            Image.Image: 標註後的圖片
        """
        try:
            draw = ImageDraw.Draw(img)
            
            # 設定字體（嘗試使用系統字體，失敗則使用預設）
            try:
                font_size = max(FONT_SIZE_MIN, img.width // FONT_SIZE_DIVISOR)
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # 設定顏色
            if is_fire:
                text_color = COLOR_FIRE_RED  # 紅色
                bg_color = COLOR_FIRE_RED + (ALPHA_SEMI_TRANSPARENT,)  # 半透明紅色
                result_text = "🔥 火災警告"
            else:
                text_color = COLOR_SAFE_GREEN  # 綠色
                bg_color = COLOR_SAFE_GREEN + (ALPHA_SEMI_TRANSPARENT,)  # 半透明綠色
                result_text = "✅ 安全"
            
            # 計算文字位置
            text_width = draw.textlength(result_text, font=font)
            text_height = font_size
            
            # 在圖片左上角繪製背景框
            padding = UI_PADDING
            rect_coords = [
                (padding, padding),
                (padding + text_width + padding * 2, padding + text_height + padding * 2)
            ]
            
            # 繪製半透明背景
            overlay = Image.new('RGBA', img.size, COLOR_WHITE + (0,))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(rect_coords, fill=bg_color)
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            
            # 重新建立 draw 物件
            draw = ImageDraw.Draw(img)
            
            # 繪製主要結果文字
            draw.text((padding * 2, padding * 2), result_text, 
                     fill=text_color, font=font)
            
            # 繪製機率資訊
            prob_text = f"火災: {p_fire:.1%} | 安全: {p_no:.1%}"
            prob_y = padding * 3 + text_height
            draw.text((padding * 2, prob_y), prob_text, 
                     fill=COLOR_WHITE, font=font)
            
            return img
            
        except Exception as e:
            logger.error(f"添加標註時發生錯誤: {e}")
            return img
    
    def resize_image(self, image_path: str, max_size: Tuple[int, int] = IMAGE_MAX_SIZE) -> str:
        """
        調整圖片大小
        
        Args:
            image_path: 圖片路徑
            max_size: 最大尺寸 (寬, 高)
            
        Returns:
            str: 調整後的圖片路徑
        """
        try:
            img = Image.open(image_path)
            
            # 如果圖片已經符合尺寸要求，直接返回
            if img.width <= max_size[0] and img.height <= max_size[1]:
                return image_path
            
            # 保持比例調整大小
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 生成新檔名
            base_name, ext = os.path.splitext(image_path)
            resized_path = f"{base_name}_resized{ext}"
            
            # 保存調整後的圖片
            img.save(resized_path, quality=IMAGE_QUALITY_HIGH)
            
            logger.info(f"圖片調整完成: {resized_path}")
            return resized_path
            
        except Exception as e:
            logger.error(f"調整圖片大小時發生錯誤: {e}")
            return image_path
    
    def get_image_info(self, image_path: str) -> Optional[dict]:
        """
        獲取圖片資訊
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            Optional[dict]: 圖片資訊或 None
        """
        try:
            if not os.path.exists(image_path):
                return None
            
            img = Image.open(image_path)
            file_size = os.path.getsize(image_path)
            
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'file_size': file_size,
                'file_size_mb': round(file_size / BYTES_TO_MB, 2)
            }
            
        except Exception as e:
            logger.error(f"獲取圖片資訊時發生錯誤: {e}")
            return None
    
    def cleanup_old_images(self, max_age_hours: int = IMAGE_CLEANUP_MAX_AGE_HOURS, max_files: int = IMAGE_CLEANUP_MAX_FILES):
        """
        清理舊圖片檔案
        
        Args:
            max_age_hours: 檔案最大年齡（小時）
            max_files: 最大檔案數量
        """
        try:
            cleanup_old_files(self.upload_folder, max_age_hours, max_files)
            logger.info(f"圖片清理完成，清理目錄: {self.upload_folder}")
        except Exception as e:
            logger.error(f"清理圖片時發生錯誤: {e}")
    
    def convert_to_web_format(self, image_path: str) -> str:
        """
        轉換圖片為網頁友好格式 (JPEG)
        
        Args:
            image_path: 原始圖片路徑
            
        Returns:
            str: 轉換後的圖片路徑
        """
        try:
            img = Image.open(image_path).convert('RGB')
            
            # 生成 JPEG 檔名
            base_name = os.path.splitext(image_path)[0]
            jpeg_path = f"{base_name}.jpg"
            
            # 保存為 JPEG
            img.save(jpeg_path, 'JPEG', quality=IMAGE_QUALITY_MEDIUM, optimize=True)
            
            logger.info(f"圖片轉換完成: {jpeg_path}")
            return jpeg_path
            
        except IOError as e:
            logger.error(f"圖片讀寫錯誤: {e}")
            return image_path
        except ValueError as e:
            logger.error(f"圖片格式錯誤: {e}")
            return image_path
        except Exception as e:
            logger.error(f"轉換圖片格式時發生意外錯誤: {e}")
            return image_path

# 全域圖片服務實例
image_service = ImageService()