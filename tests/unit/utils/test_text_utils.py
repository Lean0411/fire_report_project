"""
測試 text_utils 模組
"""
import pytest
from utils.text_utils import (
    truncate_text, remove_html_tags, normalize_whitespace,
    extract_numbers, is_valid_email, mask_sensitive_info,
    count_words, capitalize_sentences, remove_special_chars,
    filter_refusal_responses
)


class TestTextUtils:
    """TextUtils 測試類"""
    
    def test_truncate_text_normal(self):
        """測試截斷文字 - 正常情況"""
        text = "This is a long text that needs to be truncated"
        result = truncate_text(text, max_length=20)
        assert result == "This is a long te..."
        assert len(result) == 20
    
    def test_truncate_text_short(self):
        """測試截斷文字 - 短文字"""
        text = "Short text"
        result = truncate_text(text, max_length=20)
        assert result == "Short text"
    
    def test_truncate_text_exact_length(self):
        """測試截斷文字 - 剛好長度"""
        text = "Exactly twenty chars"
        result = truncate_text(text, max_length=20)
        assert result == "Exactly twenty chars"
    
    def test_truncate_text_custom_suffix(self):
        """測試截斷文字 - 自定義後綴"""
        text = "This is a long text"
        result = truncate_text(text, max_length=15, suffix=">>>")
        assert result == "This is a lo>>>"
    
    def test_remove_html_tags(self):
        """測試移除 HTML 標籤"""
        assert remove_html_tags("<p>Hello World</p>") == "Hello World"
        assert remove_html_tags("<b>Bold</b> and <i>italic</i>") == "Bold and italic"
        assert remove_html_tags("No HTML here") == "No HTML here"
        assert remove_html_tags("<script>alert('XSS')</script>") == "alert('XSS')"
        assert remove_html_tags("") == ""
    
    def test_normalize_whitespace(self):
        """測試正規化空白字符"""
        assert normalize_whitespace("  Hello   World  ") == "Hello World"
        assert normalize_whitespace("Line1\n\nLine2") == "Line1 Line2"
        assert normalize_whitespace("\t\tTabbed\t\t") == "Tabbed"
        assert normalize_whitespace("   ") == ""
        assert normalize_whitespace("NoSpaces") == "NoSpaces"
    
    def test_extract_numbers(self):
        """測試提取數字"""
        assert extract_numbers("Price is $123.45") == [123.45]
        assert extract_numbers("Room 404 and 505") == [404, 505]
        assert extract_numbers("No numbers here") == []
        assert extract_numbers("3.14159 is pi") == [3.14159]
        assert extract_numbers("-10 and +20") == [-10, 20]
    
    def test_is_valid_email(self):
        """測試驗證電子郵件"""
        # 有效的電子郵件
        assert is_valid_email("user@example.com") is True
        assert is_valid_email("test.user@domain.co.uk") is True
        assert is_valid_email("name+tag@example.org") is True
        
        # 無效的電子郵件
        assert is_valid_email("invalid.email") is False
        assert is_valid_email("@example.com") is False
        assert is_valid_email("user@") is False
        assert is_valid_email("user @example.com") is False
        assert is_valid_email("") is False
    
    def test_mask_sensitive_info(self):
        """測試遮罩敏感資訊"""
        assert mask_sensitive_info("user@example.com", "email") == "u***@example.com"
        assert mask_sensitive_info("1234567890", "phone") == "******7890"
        assert mask_sensitive_info("4111111111111111", "credit_card") == "************1111"
        assert mask_sensitive_info("short", "email") == "s***t"
        assert mask_sensitive_info("ab", "phone") == "**"
    
    def test_count_words(self):
        """測試計算字數"""
        assert count_words("Hello world") == 2
        assert count_words("  Multiple   spaces  ") == 2
        assert count_words("One") == 1
        assert count_words("") == 0
        assert count_words("   ") == 0
        assert count_words("Hello, world! How are you?") == 5
    
    def test_capitalize_sentences(self):
        """測試句子首字母大寫"""
        assert capitalize_sentences("hello. world.") == "Hello. World."
        assert capitalize_sentences("hello! how are you?") == "Hello! How are you?"
        assert capitalize_sentences("ALREADY CAPS.") == "Already caps."
        assert capitalize_sentences("no punctuation") == "No punctuation"
        assert capitalize_sentences("") == ""
    
    def test_remove_special_chars(self):
        """測試移除特殊字符"""
        assert remove_special_chars("Hello@World#2024") == "HelloWorld2024"
        assert remove_special_chars("test_file-name.txt") == "testfilenametxt"
        assert remove_special_chars("NoSpecialChars123") == "NoSpecialChars123"
        assert remove_special_chars("!@#$%^&*()") == ""
        assert remove_special_chars("") == ""
    
    def test_filter_refusal_responses_clean(self):
        """測試過濾拒絕回應 - 乾淨內容"""
        clean_response = "這是火災應急建議：1. 保持冷靜 2. 立即撤離"
        assert filter_refusal_responses(clean_response) == clean_response
    
    def test_filter_refusal_responses_with_refusal(self):
        """測試過濾拒絕回應 - 包含拒絕"""
        refusal_response = "抱歉，我無法協助這個請求。請聯繫專業人員。"
        assert filter_refusal_responses(refusal_response) == ""
        
        refusal_response2 = "我不能提供這類資訊，但是我可以幫助其他事情。"
        assert filter_refusal_responses(refusal_response2) == ""
    
    def test_filter_refusal_responses_partial_refusal(self):
        """測試過濾拒絕回應 - 部分拒絕"""
        mixed_response = "火災發生時：1. 保持冷靜\n抱歉，我無法提供更多細節。"
        result = filter_refusal_responses(mixed_response)
        assert "保持冷靜" in result
        assert "抱歉" not in result
    
    def test_filter_refusal_responses_edge_cases(self):
        """測試過濾拒絕回應 - 邊界情況"""
        assert filter_refusal_responses("") == ""
        assert filter_refusal_responses(None) == ""
        assert filter_refusal_responses("   ") == "   "
    
    def test_text_utils_unicode_handling(self):
        """測試 Unicode 處理"""
        unicode_text = "Hello 世界 🔥"
        assert truncate_text(unicode_text, 10) == "Hello 世..."
        assert count_words(unicode_text) == 2
        assert normalize_whitespace(f"  {unicode_text}  ") == unicode_text
    
    def test_text_utils_empty_inputs(self):
        """測試空輸入"""
        assert truncate_text("", 10) == ""
        assert remove_html_tags("") == ""
        assert normalize_whitespace("") == ""
        assert extract_numbers("") == []
        assert is_valid_email("") is False
        assert count_words("") == 0
        assert capitalize_sentences("") == ""
        assert remove_special_chars("") == ""