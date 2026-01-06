"""
翻译功能模块
"""
import json
import os
from typing import Dict, Any, Optional

class Translator:
    """翻译器类"""

    def __init__(self, lang_code: str = "en"):
        self.current_language = lang_code
        self.translations = self._load_translations(lang_code)

    def _load_translations(self, lang_code: str) -> Dict[str, Any]:
        """加载翻译文件"""
        try:
            lang_file = os.path.join("i18n", f"{lang_code}.json")
            with open(lang_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # 回退到默认语言
            default_file = os.path.join("i18n", "en.json")
            with open(default_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load translations: {str(e)}")
            return {}

    def translate(self, key: str, **kwargs) -> str:
        """
        翻译函数支持格式化字符串

        Args:
            key: 翻译键，支持嵌套如 "messages.connecting"
            **kwargs: 格式化参数

        Returns:
            翻译后的字符串
        """
        keys = key.split(".")
        value = self.translations

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return key

        if kwargs and isinstance(value, str):
            return value.format(**kwargs)
        return value

    def switch_language(self, lang_code: str) -> bool:
        """
        切换语言

        Args:
            lang_code: 语言代码（en/zh）

        Returns:
            是否切换成功
        """
        if lang_code in ["zh", "en"]:
            new_translations = self._load_translations(lang_code)
            if new_translations:
                self.translations = new_translations
                self.current_language = lang_code
                return True
        return False

    def get_current_language(self) -> str:
        """获取当前语言"""
        return self.current_language
