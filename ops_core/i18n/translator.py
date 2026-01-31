"""
Translation module
"""
import json
import os
from typing import Dict, Any, Optional

class Translator:
    """Translator class"""

    def __init__(self, lang_code: str = "en"):
        self.current_language = lang_code
        self.translations = self._load_translations(lang_code)

    def _load_translations(self, lang_code: str) -> Dict[str, Any]:
        """Load translation file"""
        try:
            lang_file = os.path.join("i18n", f"{lang_code}.json")
            with open(lang_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to default language
            default_file = os.path.join("i18n", "en.json")
            with open(default_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load translations: {str(e)}")
            return {}

    def translate(self, key: str, **kwargs) -> str:
        """
        Translate function supports formatted strings

        Args:
            key: Translation key, supports nested like "messages.connecting"
            **kwargs: Formatting parameters

        Returns:
            Translated string
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
        Switch language

        Args:
            lang_code: Language code (en/zh)

        Returns:
            Whether switch was successful
        """
        if lang_code in ["zh", "en"]:
            new_translations = self._load_translations(lang_code)
            if new_translations:
                self.translations = new_translations
                self.current_language = lang_code
                return True
        return False

    def get_current_language(self) -> str:
        """Get current language"""
        return self.current_language
