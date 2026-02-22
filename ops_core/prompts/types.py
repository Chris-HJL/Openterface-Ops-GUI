"""
Scene type enumeration
"""
from enum import Enum


class SceneType(Enum):
    """Scene type enumeration"""
    AUTO = "auto"
    GENERAL = "general"
    BIOS = "bios"
    WINDOWS = "windows"
    LINUX = "linux"
    OS_INSTALLATION = "os_installation"
