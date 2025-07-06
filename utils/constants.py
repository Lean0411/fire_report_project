"""
常數定義模組
包含應用程式中使用的常數和靜態數據
"""
from config.constants import MAX_FILE_SIZE, HTTP_OK, HTTP_BAD_REQUEST, HTTP_INTERNAL_ERROR

# 類別標籤對應表（用於前端顯示）
CATEGORY_LABELS = {
    'general': '一般民眾',
    'firefighter': '消防隊員', 
    'management': '管理單位',
    'emergency_action_plan': '緊急行動計劃',
    'evacuation_procedures': '疏散程序',
    'evacuation_preparedness': '疏散準備',
    'shelter_in_place': '就地避難',
    'communication_protocol': '通訊協議',
    'initial_assessment': '初步評估',
    'suppression_strategy': '滅火策略',
    'safety_protocols': '安全協議',
    'emergency_management_protocols': '緊急管理協議',
    'resource_allocation': '資源配置',
    'public_communication': '公眾溝通',
    'incident_command': '事故指揮',
    'tactical_operations': '戰術行動',
    'personnel_safety': '人員安全',
    'communication_coordination': '溝通協調',
    'resilience_and_recovery': '復原與重建'
}

# 允許的檔案擴展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 使用者角色
USER_ROLES = {
    'general': '一般民眾',
    'firefighter': '消防隊員',
    'management': '管理單位'
}

# AI 提供者
AI_PROVIDERS = {
    'openai': 'OpenAI GPT',
    'ollama': 'Ollama Local LLM'
}

# 檔案大小限制 (bytes) - 從 config.constants 引入
# MAX_FILE_SIZE 已經從 config.constants 引入

# HTTP 狀態碼
HTTP_STATUS = {
    'OK': HTTP_OK,
    'BAD_REQUEST': HTTP_BAD_REQUEST,
    'INTERNAL_SERVER_ERROR': HTTP_INTERNAL_ERROR
}