"""
Global configuration management
"""
import os
from enum import Enum
from typing import Optional


class ScreenCaptureMode(str, Enum):
    """屏幕捕获模式枚举"""
    GETTARGETSCREEN = "gettargetscreen"
    LASTIMAGE = "lastimage"
    HYBRID = "hybrid"  # 优先使用 gettargetscreen，失败时回退到 lastimage


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
    IMAGE_SERVER_TIMEOUT: int = 30  # lastimage 使用较短超时

    # 新增：屏幕捕获专用超时 (实时捕获需要更长时间)
    SCREEN_CAPTURE_TIMEOUT: int = 120  # gettargetscreen 使用较长超时

    # 新增：屏幕捕获命令配置 (默认值，之后会被环境变量覆盖)
    SCREEN_CAPTURE_MODE: ScreenCaptureMode = ScreenCaptureMode.HYBRID

    # 新增：是否启用分辨率记录
    RECORD_SCREEN_RESOLUTION: bool = True

    # 新增：屏幕分辨率配置（用于坐标转换）
    SCREEN_WIDTH: int = 1920
    SCREEN_HEIGHT: int = 1080

    # 新增：坐标系统配置
    COORD_SYSTEM: str = "hid"  # "hid" 或 "pixel"，默认使用 HID 归一化坐标 (0-4096)

    IMAGES_DIR: str = "./images"
    OUTPUT_DIR: str = "./output"

    # 会话配置
    DEFAULT_LANGUAGE: str = "en"
    MAX_REACT_ITERATIONS: int = 20

    # 清理配置
    TASK_TTL_SECONDS: int = 3600
    SESSION_TTL_SECONDS: int = 7200
    MAX_TASKS: int = 100
    MAX_SESSIONS: int = 50
    CLEANUP_INTERVAL_SECONDS: int = 300

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

    @classmethod
    def reload_from_env(cls):
        """从环境变量重新加载配置"""
        # 重新加载屏幕捕获模式
        cls.SCREEN_CAPTURE_MODE = cls._load_screen_capture_mode()
        # 重新加载屏幕捕获超时
        cls.SCREEN_CAPTURE_TIMEOUT = cls._load_screen_capture_timeout()
        # 重新加载屏幕分辨率
        cls.SCREEN_WIDTH = cls._load_screen_width()
        cls.SCREEN_HEIGHT = cls._load_screen_height()
        # 重新加载坐标系统
        cls.COORD_SYSTEM = cls._load_coord_system()

    @classmethod
    def _load_screen_capture_mode(cls) -> ScreenCaptureMode:
        """从环境变量获取屏幕捕获模式"""
        mode_str = cls.get_env("SCREEN_CAPTURE_MODE", "hybrid").lower()
        try:
            return ScreenCaptureMode(mode_str)
        except ValueError:
            print(f"Warning: Invalid SCREEN_CAPTURE_MODE '{mode_str}', using default 'hybrid'")
            return ScreenCaptureMode.HYBRID

    @classmethod
    def _load_screen_capture_timeout(cls) -> int:
        """从环境变量获取屏幕捕获超时"""
        try:
            return int(cls.get_env("SCREEN_CAPTURE_TIMEOUT", "120"))
        except ValueError:
            return 120

    @classmethod
    def _load_screen_width(cls) -> int:
        """从环境变量获取屏幕宽度"""
        try:
            return int(cls.get_env("SCREEN_WIDTH", "1920"))
        except ValueError:
            print("Warning: Invalid SCREEN_WIDTH, using default 1920")
            return 1920

    @classmethod
    def _load_screen_height(cls) -> int:
        """从环境变量获取屏幕高度"""
        try:
            return int(cls.get_env("SCREEN_HEIGHT", "1080"))
        except ValueError:
            print("Warning: Invalid SCREEN_HEIGHT, using default 1080")
            return 1080

    @classmethod
    def _load_coord_system(cls) -> str:
        """从环境变量获取坐标系统"""
        system = cls.get_env("COORD_SYSTEM", "hid").lower()
        if system in ["hid", "pixel"]:
            return system
        print(f"Warning: Invalid COORD_SYSTEM '{system}', using default 'hid'")
        return "hid"


# 在类定义后从环境变量加载配置
Config.reload_from_env()
