"""
完整键映射表
映射 LLM 输出的键名到 TCP 命令格式
"""

from typing import Dict, Set

# 标准键映射
KEY_MAP: Dict[str, str] = {
    # 标准控制键
    "Enter": "{Enter}",
    "Tab": "{Tab}",
    "Escape": "{Escape}",
    "Esc": "{Escape}",  # 别名
    "Backspace": "{Backspace}",
    "BS": "{Backspace}",  # 别名
    "Delete": "{Delete}",
    "Del": "{Delete}",  # 别名
    "Insert": "{Insert}",
    "Ins": "{Insert}",  # 别名
    "Home": "{Home}",
    "End": "{End}",
    "PageUp": "{PgUp}",
    "PgUp": "{PgUp}",  # 别名
    "PageDown": "{PgDn}",
    "PgDn": "{PgDn}",  # 别名

    # 方向键
    "Up": "{Up}",
    "Down": "{Down}",
    "Left": "{Left}",
    "Right": "{Right}",

    # 功能键 F1-F12
    "F1": "{F1}", "F2": "{F2}", "F3": "{F3}", "F4": "{F4}",
    "F5": "{F5}", "F6": "{F6}", "F7": "{F7}", "F8": "{F8}",
    "F9": "{F9}", "F10": "{F10}", "F11": "{F11}", "F12": "{F12}",

    # 控制键（注意：不支持组合，仅单个按键）
    "Ctrl": "{Ctrl}",
    "Alt": "{Alt}",
    "Shift": "{Shift}",
    "Win": "{Win}",

    # 其他特殊键
    "Space": "{Space}",
    "PrintScreen": "{PrintScreen}",
    "ScrollLock": "{ScrollLock}",
    "Pause": "{Pause}",

    # 常见缩写
    "OK": "{Enter}",
    "Cancel": "{Escape}",
}

# 特殊键名称列表（用于验证）
VALID_SPECIAL_KEYS: Set[str] = set(KEY_MAP.keys())

# 描述性文档
KEY_DESCRIPTIONS: Dict[str, str] = {
    "Enter": "回车键",
    "Tab": "制表键（切换焦点）",
    "Escape": "退出键",
    "Backspace": "退格键（删除光标前字符）",
    "Delete": "删除键（删除光标后字符）",
    "Up/Down/Left/Right": "方向键",
    "F1-F12": "功能键",
    "Ctrl/Alt/Shift": "修饰键（注意：不支持组合键）",
    "Home/End": "跳转到行首/行尾",
    "PageUp/PageDown": "上页/下页",
    "Space": "空格键",
}


def normalize_key_name(key_name: str) -> str:
    """
    标准化键名（处理大小写和别名）

    Args:
        key_name: 原始键名，如 'enter', 'ENTER', 'Tab'

    Returns:
        标准化键名，如 'Enter', 'Tab'

    Example:
        >>> normalize_key_name("enter")
        "Enter"
        >>> normalize_key_name("ESC")
        "Esc"
    """
    if not key_name:
        return ""

    # 先转为首字母大写
    normalized = key_name[0].upper() + key_name[1:].lower() if key_name else ""

    # 查找映射表
    if normalized in KEY_MAP:
        return normalized

    # 尝试全大写查找（用于 F1, F2 等）
    upper = key_name.upper()
    if upper in KEY_MAP:
        return upper

    # 返回原样（调用方会处理未找到的情况）
    return key_name


def is_special_key(key_name: str) -> bool:
    """
    检查是否为特殊键

    Args:
        key_name: 键名

    Returns:
        True 如果是特殊键（在 KEY_MAP 中）
    """
    return normalize_key_name(key_name) in VALID_SPECIAL_KEYS


def get_tcp_key_code(key_name: str) -> str:
    """
    获取 TCP 命令格式的键码

    Args:
        key_name: 键名

    Returns:
        如 "{Enter}", "{F5}" 等

    Raises:
        KeyError: 键名不在映射表中

    Example:
        >>> get_tcp_key_code("Enter")
        "{Enter}"
        >>> get_tcp_key_code("F5")
        "{F5}"
    """
    normalized = normalize_key_name(key_name)
    if normalized not in KEY_MAP:
        raise KeyError(f"Unknown key: {key_name}. "
                      f"Valid keys: {', '.join(sorted(VALID_SPECIAL_KEYS))}")
    return KEY_MAP[normalized]


if __name__ == "__main__":
    # 简单测试
    print("Key Map Tests:")
    print(f"  normalize_key_name('enter') = {normalize_key_name('enter')}")
    print(f"  normalize_key_name('ESC') = {normalize_key_name('ESC')}")
    print(f"  is_special_key('Enter') = {is_special_key('Enter')}")
    print(f"  is_special_key('InvalidKey') = {is_special_key('InvalidKey')}")
    print(f"  get_tcp_key_code('Enter') = {get_tcp_key_code('Enter')}")
    print(f"  get_tcp_key_code('F5') = {get_tcp_key_code('F5')}")
