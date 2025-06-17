"""
SOP服務模組
提供標準作業程序(SOP)知識庫管理和角色建議功能
"""
import json
import os
from typing import Dict, Any, Optional
from functools import lru_cache

from config.settings import Config
from config.logging_config import get_logger
from utils.constants import USER_ROLES

logger = get_logger(__name__)

class SOPService:
    """SOP服務類，負責知識庫管理和角色建議"""
    
    def __init__(self):
        self.knowledge_base_path = Config.KNOWLEDGE_BASE_PATH
        self._sop_data: Optional[Dict] = None
    
    @lru_cache(maxsize=1)
    def load_sop_recommendations(self) -> Dict[str, Any]:
        """
        載入SOP建議內容（帶快取）
        
        Returns:
            Dict[str, Any]: SOP知識庫數據
        """
        try:
            if not os.path.exists(self.knowledge_base_path):
                logger.error(f"SOP知識庫檔案不存在: {self.knowledge_base_path}")
                return {}
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info("SOP知識庫載入成功")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"SOP知識庫JSON格式錯誤: {e}")
            return {}
        except Exception as e:
            logger.error(f"無法載入SOP知識庫: {e}")
            return {}
    
    def get_role_recommendations(self, role: str, is_fire: bool) -> Dict[str, Any]:
        """
        根據角色和火災狀況獲取建議
        
        Args:
            role: 使用者角色 (general/firefighter/management)
            is_fire: 是否檢測到火災
            
        Returns:
            Dict[str, Any]: 角色對應的建議內容
        """
        sop_data = self.load_sop_recommendations()
        
        if not sop_data or role not in sop_data:
            logger.warning(f"角色 '{role}' 不存在於SOP知識庫中")
            return {}
        
        role_sop = sop_data[role]
        recommendations = {}
        
        if is_fire:
            # 火災情況下，提供所有相關建議
            recommendations = role_sop.copy()
            logger.info(f"為角色 '{role}' 提供火災應急建議")
        else:
            # 非火災情況下，只提供預防性建議
            if role == "general":
                if "emergency_action_plan" in role_sop:
                    recommendations["emergency_action_plan"] = role_sop["emergency_action_plan"]
            elif role == "firefighter":
                if "initial_assessment" in role_sop:
                    recommendations["initial_assessment"] = role_sop["initial_assessment"]
            elif role == "management":
                if "emergency_management_protocols" in role_sop:
                    recommendations["emergency_management_protocols"] = role_sop["emergency_management_protocols"]
            
            logger.info(f"為角色 '{role}' 提供預防性建議")
        
        return recommendations
    
    def get_all_recommendations_for_role(self, role: str) -> Dict[str, Any]:
        """
        獲取特定角色的所有建議
        
        Args:
            role: 使用者角色
            
        Returns:
            Dict[str, Any]: 該角色的所有建議
        """
        sop_data = self.load_sop_recommendations()
        return sop_data.get(role, {})
    
    def get_available_roles(self) -> Dict[str, str]:
        """
        獲取可用的角色列表
        
        Returns:
            Dict[str, str]: 角色代碼到顯示名稱的映射
        """
        sop_data = self.load_sop_recommendations()
        available_roles = {}
        
        for role_code in sop_data.keys():
            if role_code in USER_ROLES:
                available_roles[role_code] = USER_ROLES[role_code]
        
        return available_roles
    
    def get_recommendation_categories(self, role: str) -> list[str]:
        """
        獲取特定角色的建議類別
        
        Args:
            role: 使用者角色
            
        Returns:
            list[str]: 建議類別列表
        """
        role_data = self.get_all_recommendations_for_role(role)
        return list(role_data.keys())
    
    def validate_sop_data(self) -> Dict[str, Any]:
        """
        驗證SOP數據的完整性
        
        Returns:
            Dict[str, Any]: 驗證結果
        """
        sop_data = self.load_sop_recommendations()
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if not sop_data:
            validation_result['is_valid'] = False
            validation_result['errors'].append("SOP數據為空或載入失敗")
            return validation_result
        
        # 檢查必要角色
        required_roles = set(USER_ROLES.keys())
        available_roles = set(sop_data.keys())
        
        missing_roles = required_roles - available_roles
        if missing_roles:
            validation_result['warnings'].append(f"缺少角色定義: {missing_roles}")
        
        # 統計信息
        validation_result['statistics'] = {
            'total_roles': len(sop_data),
            'total_categories': sum(len(role_data) for role_data in sop_data.values()),
            'roles': list(sop_data.keys())
        }
        
        # 檢查每個角色的數據結構
        for role, role_data in sop_data.items():
            if not isinstance(role_data, dict):
                validation_result['errors'].append(f"角色 '{role}' 的數據格式錯誤")
                validation_result['is_valid'] = False
                continue
            
            for category, actions in role_data.items():
                if not isinstance(actions, list):
                    validation_result['warnings'].append(
                        f"角色 '{role}' 的類別 '{category}' 不是列表格式"
                    )
                elif not actions:
                    validation_result['warnings'].append(
                        f"角色 '{role}' 的類別 '{category}' 為空"
                    )
        
        logger.info(f"SOP數據驗證完成: {'通過' if validation_result['is_valid'] else '失敗'}")
        return validation_result
    
    def reload_sop_data(self):
        """重新載入SOP數據（清除快取）"""
        self.load_sop_recommendations.cache_clear()
        logger.info("SOP數據快取已清除，下次訪問將重新載入")
    
    def get_sop_file_info(self) -> Dict[str, Any]:
        """
        獲取SOP檔案資訊
        
        Returns:
            Dict[str, Any]: 檔案資訊
        """
        try:
            if not os.path.exists(self.knowledge_base_path):
                return {'exists': False}
            
            stat = os.stat(self.knowledge_base_path)
            return {
                'exists': True,
                'path': self.knowledge_base_path,
                'size': stat.st_size,
                'modified_time': stat.st_mtime,
                'readable': os.access(self.knowledge_base_path, os.R_OK)
            }
        except Exception as e:
            logger.error(f"獲取SOP檔案資訊失敗: {e}")
            return {'exists': False, 'error': str(e)}

# 全域SOP服務實例
sop_service = SOPService()