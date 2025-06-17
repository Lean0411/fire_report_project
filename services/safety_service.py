"""
安全服務模組
提供火災安全建議和緊急處置指南
"""
from typing import Dict, List, Any
from datetime import datetime

from config.logging_config import get_logger
from utils.constants import USER_ROLES

logger = get_logger(__name__)

class SafetyService:
    """安全服務類，提供火災安全建議和指南"""
    
    def __init__(self):
        self.safety_tips = self._initialize_safety_tips()
    
    def _initialize_safety_tips(self) -> Dict[str, List[str]]:
        """初始化安全建議數據"""
        return {
            "immediate_actions": [
                "立即撥打119報警，說明火災位置和規模",
                "如在室內，請低身爬行避免濃煙，用濕毛巾掩住口鼻",
                "觸摸門把前先檢查溫度，如燙手請勿開門",
                "若火勢小且安全，可使用滅火器撲滅，記住「拉、瞄、壓、掃」四步驟",
                "確保逃生路線暢通，不可使用電梯"
            ],
            "evacuation_tips": [
                "選擇最近的安全出口，避開濃煙區域",
                "幫助行動不便者一同撤離",
                "到達安全地點後立即清點人數",
                "不要返回火場搶救物品",
                "在安全地點等待消防人員到達"
            ],
            "prevention_measures": [
                "定期檢查電線和電器設備，避免過載使用",
                "保持逃生通道暢通，不堆放雜物",
                "家中準備滅火器、煙霧偵測器和逃生繩",
                "制定家庭火災逃生計畫並定期演練",
                "廚房用火後務必檢查爐火是否完全熄滅"
            ],
            "high_rise_building": [
                "熟悉大樓緊急逃生路線和集合點",
                "火災時絕對不可使用電梯",
                "若煙霧瀰漫樓梯間，考慮就地避難等待救援",
                "在安全樓層等待消防人員指示",
                "使用濕毛巾密封門縫防止煙霧進入"
            ],
            "outdoor_fire": [
                "立即遠離火源，尋找安全地點",
                "如被火包圍，選擇植被稀少的地方突圍",
                "用水浸濕衣物覆蓋身體",
                "避免在山脊、山頂等地勢高的地方停留",
                "聽從消防人員和相關部門的疏散指示"
            ],
            "vehicle_fire": [
                "立即停車熄火，切斷電源",
                "迅速離開車輛，撤離到安全距離",
                "如車門無法開啟，擊破車窗逃生",
                "撥打119報警並告知具體位置",
                "不要試圖搶救車內物品"
            ]
        }
    
    def generate_fire_safety_tips(self) -> Dict[str, List[str]]:
        """
        生成火災安全建議
        
        Returns:
            Dict[str, List[str]]: 安全建議字典
        """
        logger.info("生成火災安全建議")
        return self.safety_tips.copy()
    
    def get_situation_specific_advice(self, situation: str) -> List[str]:
        """
        獲取特定情況的安全建議
        
        Args:
            situation: 情況類型
            
        Returns:
            List[str]: 該情況的安全建議
        """
        advice = self.safety_tips.get(situation, [])
        if advice:
            logger.info(f"提供 '{situation}' 情況的安全建議")
        else:
            logger.warning(f"未找到 '{situation}' 情況的安全建議")
        
        return advice
    
    def get_emergency_contacts(self) -> Dict[str, str]:
        """
        獲取緊急聯絡方式
        
        Returns:
            Dict[str, str]: 緊急聯絡資訊
        """
        return {
            "消防局": "119",
            "警察局": "110", 
            "救護車": "119",
            "災害應變中心": "1999",
            "瓦斯公司緊急電話": "1999",
            "台電客服": "1911"
        }
    
    def generate_role_based_advice(self, role: str, is_fire: bool, 
                                 fire_probability: float = 0.0) -> Dict[str, Any]:
        """
        根據角色生成個性化安全建議
        
        Args:
            role: 使用者角色
            is_fire: 是否檢測到火災
            fire_probability: 火災機率
            
        Returns:
            Dict[str, Any]: 個性化建議
        """
        advice = {
            'role': role,
            'role_name': USER_ROLES.get(role, '未知角色'),
            'is_fire': is_fire,
            'fire_probability': fire_probability,
            'timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'priority_actions': [],
            'emergency_contacts': self.get_emergency_contacts()
        }
        
        if role == 'general':
            advice.update(self._get_general_public_advice(is_fire, fire_probability))
        elif role == 'firefighter':
            advice.update(self._get_firefighter_advice(is_fire, fire_probability))
        elif role == 'management':
            advice.update(self._get_management_advice(is_fire, fire_probability))
        else:
            logger.warning(f"未知角色: {role}")
            advice['recommendations'] = self.safety_tips['immediate_actions']
        
        logger.info(f"為角色 '{role}' 生成個性化建議")
        return advice
    
    def _get_general_public_advice(self, is_fire: bool, fire_probability: float) -> Dict[str, Any]:
        """獲取一般民眾建議"""
        if is_fire or fire_probability > 0.7:
            return {
                'recommendations': self.safety_tips['immediate_actions'] + 
                                 self.safety_tips['evacuation_tips'],
                'priority_actions': [
                    "立即撥打119報警",
                    "確保人身安全，迅速撤離",
                    "協助他人安全疏散"
                ]
            }
        else:
            return {
                'recommendations': self.safety_tips['prevention_measures'],
                'priority_actions': [
                    "檢查周圍環境安全",
                    "確認火災預防措施",
                    "準備緊急應變計畫"
                ]
            }
    
    def _get_firefighter_advice(self, is_fire: bool, fire_probability: float) -> Dict[str, Any]:
        """獲取消防隊員建議"""
        if is_fire or fire_probability > 0.5:
            return {
                'recommendations': [
                    "立即評估火場狀況和危險程度",
                    "確認人員搜救需求",
                    "評估灭火戰術和資源需求",
                    "建立指揮系統和通訊頻道",
                    "協調其他緊急服務單位"
                ],
                'priority_actions': [
                    "現場安全評估",
                    "人命搜救優先",
                    "建立火場指揮"
                ]
            }
        else:
            return {
                'recommendations': [
                    "進行現場風險評估",
                    "檢查潛在火災隱患",
                    "確認滅火設備狀態",
                    "評估預防措施效果"
                ],
                'priority_actions': [
                    "風險評估",
                    "設備檢查",
                    "預防宣導"
                ]
            }
    
    def _get_management_advice(self, is_fire: bool, fire_probability: float) -> Dict[str, Any]:
        """獲取管理單位建議"""
        if is_fire or fire_probability > 0.6:
            return {
                'recommendations': [
                    "啟動緊急應變中心",
                    "協調跨部門應變資源",
                    "發布公眾警報和疏散指示",
                    "媒體溝通和資訊發布",
                    "後續復原計畫準備"
                ],
                'priority_actions': [
                    "啟動應變中心",
                    "資源協調調度",
                    "公眾溝通管理"
                ]
            }
        else:
            return {
                'recommendations': [
                    "檢討應變計畫完整性",
                    "評估預防措施效果",
                    "加強防火宣導教育",
                    "定期演練和訓練"
                ],
                'priority_actions': [
                    "政策檢討",
                    "預防宣導",
                    "教育訓練"
                ]
            }
    
    def get_safety_checklist(self, environment: str = "general") -> List[str]:
        """
        獲取安全檢查清單
        
        Args:
            environment: 環境類型 (general/home/office/factory)
            
        Returns:
            List[str]: 安全檢查項目
        """
        checklists = {
            "general": [
                "檢查滅火器位置和有效期限",
                "確認逃生路線暢通",
                "測試煙霧探測器功能",
                "檢查電線和插座狀況",
                "確認緊急出口標示清楚"
            ],
            "home": [
                "廚房爐具使用後確實關閉",
                "電器用品使用後拔除插頭", 
                "陽台和樓梯間不堆放雜物",
                "準備手電筒和逃生繩",
                "制定家庭逃生計畫"
            ],
            "office": [
                "辦公設備定期維護檢查",
                "緊急照明系統正常運作",
                "防火門保持關閉狀態",
                "員工熟悉疏散程序",
                "定期舉行消防演練"
            ],
            "factory": [
                "危險物品妥善儲存標示",
                "消防設備定期檢測維護",
                "員工接受消防安全訓練",
                "緊急停機程序建立完整",
                "與消防單位建立聯繫機制"
            ]
        }
        
        return checklists.get(environment, checklists["general"])

# 全域安全服務實例
safety_service = SafetyService()