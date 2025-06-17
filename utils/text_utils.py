"""
文字處理工具模組
提供文字過濾、清理和處理相關功能
"""
import re
from typing import Optional

def filter_refusal_responses(text: str) -> str:
    """
    過濾掉AI的拒絕回應，只保留有用的內容
    
    Args:
        text: 待過濾的文字
        
    Returns:
        str: 過濾後的文字
    """
    if not text or not text.strip():
        return ""
    
    # 定義拒絕回應的關鍵詞和模式
    refusal_patterns = [
        "抱歉，我無法識別或描述圖片中的具體內容",
        "我無法識別或描述圖片中的具體內容",
        "很抱歉，我無法識別或描述圖片中的人物或具體細節",
        "我無法識別或描述圖片中的人物或具體細節",
        "抱歉，我無法識別圖片",
        "我無法識別圖片",
        "很抱歉，我無法識別圖片",
        "抱歉，我無法分析圖片",
        "我無法分析圖片",
        "很抱歉，我無法分析圖片",
        "我無法查看或分析圖片",
        "抱歉，我無法查看或分析圖片",
        "很抱歉，我無法查看或分析圖片",
        "我無法直接識別圖片內容",
        "抱歉，我無法直接識別圖片內容",
        "很抱歉，我無法直接識別圖片內容",
        "我無法看到圖片",
        "抱歉，我無法看到圖片",
        "很抱歉，我無法看到圖片",
        "我無法識別圖片中的人物",
        "抱歉，我無法識別圖片中的人物",
        "很抱歉，我無法識別圖片中的人物",
        "我無法描述圖片中的具體細節",
        "抱歉，我無法描述圖片中的具體細節",
        "很抱歉，我無法描述圖片中的具體細節"
    ]
    
    # 分割文本為段落
    paragraphs = text.split('\n')
    filtered_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # 檢查是否包含拒絕模式
        contains_refusal = False
        for pattern in refusal_patterns:
            if pattern in paragraph:
                contains_refusal = True
                break
        
        # 如果不包含拒絕模式，保留這個段落
        if not contains_refusal:
            filtered_paragraphs.append(paragraph)
    
    filtered_text = '\n'.join(filtered_paragraphs)
    
    # 如果過濾後的文本太短或只包含一般性建議，返回空字串
    if len(filtered_text.strip()) < 50:
        return ""
    
    # 檢查是否只包含一般性的無意義內容
    generic_phrases = [
        "不過，我可以提供一些一般的消防安全建議",
        "以下是一些一般的消防安全建議",
        "我可以提供一些基本的消防安全建議",
        "以下是一些基本的消防安全建議",
        "不過，我可以根據火災情境提供一般的消防安全建議",
        "我可以根據火災情境提供一般的消防安全建議",
        "不過，我可以提供一般性的消防安全建議",
        "我可以提供一般性的消防安全建議"
    ]
    
    # 如果包含這些一般性短語，且沒有其他具體內容，過濾掉
    for phrase in generic_phrases:
        if phrase in filtered_text and len(filtered_text.replace(phrase, "").strip()) < 100:
            return ""
    
    return filtered_text

def clean_text(text: str) -> str:
    """
    清理文字，移除多餘的空白和特殊字元
    
    Args:
        text: 待清理的文字
        
    Returns:
        str: 清理後的文字
    """
    if not text:
        return ""
    
    # 移除多餘的空白字元
    text = re.sub(r'\s+', ' ', text)
    
    # 移除開頭和結尾的空白
    text = text.strip()
    
    return text

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    截斷文字到指定長度
    
    Args:
        text: 待截斷的文字
        max_length: 最大長度
        suffix: 截斷後的後綴
        
    Returns:
        str: 截斷後的文字
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def sanitize_filename(filename: str) -> str:
    """
    清理檔案名稱，移除不安全字元
    
    Args:
        filename: 原始檔案名稱
        
    Returns:
        str: 清理後的檔案名稱
    """
    if not filename:
        return "unknown_file"
    
    # 移除不安全的字元
    safe_chars = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # 確保不以點開頭
    if safe_chars.startswith('.'):
        safe_chars = 'file_' + safe_chars
    
    return safe_chars

def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """
    從文字中提取關鍵詞
    
    Args:
        text: 待分析的文字
        max_keywords: 最大關鍵詞數量
        
    Returns:
        list: 關鍵詞列表
    """
    if not text:
        return []
    
    # 火災相關關鍵詞
    fire_keywords = [
        '火災', '火焰', '煙霧', '燃燒', '滅火', '消防', '逃生', 
        '疏散', '警報', '安全', '緊急', '救援', '危險', '高溫'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in fire_keywords:
        if keyword in text or keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords[:max_keywords]