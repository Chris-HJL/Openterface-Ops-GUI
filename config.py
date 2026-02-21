"""
Global configuration management
"""
import os
from typing import Optional

class Config:
    """应用配置类"""

    # API配置
    DEFAULT_API_URL: str = "http://localhost:11434/v1/chat/completions"
    DEFAULT_MODEL: str = "qwen3-vl:8b-thinking-q4_K_M"
    DEFAULT_UI_MODEL_API_URL: str = "http://localhost:2345/v1/chat/completions"
    DEFAULT_UI_MODEL: str = "fara-7b"

    # RAG配置
    RAG_API_BASE: str = "http://localhost:11434/v1"
    RAG_EMBED_MODEL: str = "qwen3-embedding:0.6b"
    RAG_INDEX_DIR: str = "./index"
    RAG_DOCS_DIR: str = "./docs"

    # 图像服务器配置
    IMAGE_SERVER_HOST: str = "localhost"
    IMAGE_SERVER_PORT: int = 12345
    IMAGE_SERVER_TIMEOUT: int = 120
    IMAGES_DIR: str = "./images"
    OUTPUT_DIR: str = "./output"

    # 会话配置
    DEFAULT_LANGUAGE: str = "en"
    MAX_REACT_ITERATIONS: int = 20

    @classmethod
    def get_env(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable"""
        return os.getenv(key, default)

    @classmethod
    def get_api_key(cls) -> str:
        """Get API key"""
        return cls.get_env("LLM_API_KEY", "EMPTY")

    @classmethod
    def get_ui_api_key(cls) -> str:
        """Get UI-Model API key"""
        return cls.get_env("UI_API_KEY", "EMPTY")
