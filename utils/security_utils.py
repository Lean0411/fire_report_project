"""
安全工具模組
提供輸入驗證、清理和安全檢查功能
"""
import re
import html
from typing import Any


def sanitize_string_input(value: Any, max_length: int = 100, default: str = "") -> str:
    """
    清理字串輸入，防止XSS攻擊

    Args:
        value: 輸入值
        max_length: 最大長度
        default: 預設值

    Returns:
        str: 清理後的字串
    """
    if not isinstance(value, str):
        return default

    # 移除前後空白
    value = value.strip()

    # 檢查長度
    if len(value) > max_length:
        return default

    # HTML 轉義防止 XSS
    value = html.escape(value)

    # 移除危險字符
    value = re.sub(r'[<>"\']', "", value)

    return value


def validate_role(role: Any) -> str:
    """
    驗證用戶角色

    Args:
        role: 用戶角色

    Returns:
        str: 驗證後的角色
    """
    from utils.constants import USER_ROLES

    role = sanitize_string_input(role, max_length=50, default="general")

    if role not in USER_ROLES:
        return "general"

    return role


def validate_ai_provider(provider: Any) -> str:
    """
    驗證AI提供者

    Args:
        provider: AI提供者

    Returns:
        str: 驗證後的提供者
    """
    from utils.constants import AI_PROVIDERS

    provider = sanitize_string_input(provider, max_length=20, default="openai")

    if provider not in AI_PROVIDERS:
        return "openai"

    return provider


def validate_probability(value: Any) -> float:
    """
    驗證機率值

    Args:
        value: 機率值

    Returns:
        float: 驗證後的機率值 (0.0-1.0)
    """
    if not isinstance(value, (int, float)):
        return 0.0

    value = float(value)

    if value < 0:
        return 0.0
    elif value > 1:
        return 1.0

    return value


def validate_boolean(value: Any) -> bool:
    """
    驗證布爾值

    Args:
        value: 布爾值

    Returns:
        bool: 驗證後的布爾值
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")

    return False


def validate_filename(filename: str) -> bool:
    """
    驗證檔案名是否安全

    Args:
        filename: 檔案名

    Returns:
        bool: 是否安全
    """
    if not filename or not isinstance(filename, str):
        return False

    # 檢查危險字符
    dangerous_chars = ["..", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in dangerous_chars:
        if char in filename:
            return False

    # 檢查長度
    if len(filename) > 255:
        return False

    return True


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    遮蔽敏感數據

    Args:
        data: 敏感數據
        mask_char: 遮蔽字符
        visible_chars: 可見字符數

    Returns:
        str: 遮蔽後的數據
    """
    if not data or len(data) <= visible_chars:
        return mask_char * len(data) if data else ""

    return data[:visible_chars] + mask_char * (len(data) - visible_chars)


def validate_api_key(api_key: str) -> bool:
    """
    驗證API Key格式

    Args:
        api_key: API Key

    Returns:
        bool: 是否有效
    """
    if not api_key or not isinstance(api_key, str):
        return False

    # 基本長度檢查
    if len(api_key) < 10:
        return False

    # 檢查是否只包含有效字符
    if not re.match(r"^[a-zA-Z0-9\-_\.]+$", api_key):
        return False

    return True


def validate_url(url: str) -> bool:
    """
    驗證URL格式

    Args:
        url: URL字串

    Returns:
        bool: 是否有效
    """
    if not url or not isinstance(url, str):
        return False

    # 基本URL格式檢查
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    return bool(url_pattern.match(url))
