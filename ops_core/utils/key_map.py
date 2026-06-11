"""
完整键映射表
映射 LLM 输出的键名到 TCP 命令格式
支持组合键解析 (^=Ctrl, +=Shift, !=Alt, #=Win)
"""

from typing import Dict, Set, Tuple, List

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

    # 修饰键（单独使用）
    "Ctrl": "{Ctrl}",
    "Alt": "{Alt}",
    "Shift": "{Shift}",
    "Win": "{Win}",

    # 其他特殊键
    "Space": "{Space}",
    "PrintScreen": "{PrintScreen}",
    "ScrollLock": "{ScrollLock}",
    "Pause": "{Pause}",
    "CapsLock": "{CapsLock}",
    "NumLock": "{NumLock}",

    # 括号/符号键
    "BracketLeft": "{BracketLeft}",     # [
    "BracketRight": "{BracketRight}",   # ]
    "Semicolon": "{Semicolon}",         # ;
    "Apostrophe": "{Apostrophe}",       # '
    "QuoteLeft": "{QuoteLeft}",         # `
    "Comma": "{Comma}",                 # ,
    "Period": "{Period}",               # .
    "Slash": "{Slash}",                 # /
    "Minus": "{Minus}",                 # -
    "Equal": "{Equal}",                 # =
    "Backslash": "{Backslash}",         # \

    # 常见缩写
    "OK": "{Enter}",
    "Cancel": "{Escape}",
}

# 特殊键名称列表（用于验证）
VALID_SPECIAL_KEYS: Set[str] = set(KEY_MAP.keys())

# 组合键前缀映射
COMBO_PREFIX: Dict[str, str] = {
    "^": "Ctrl",
    "+": "Shift",
    "!": "Alt",
    "#": "Win",
}

# 反向映射：修饰键名 -> 前缀符号
MODIFIER_TO_PREFIX: Dict[str, str] = {v: k for k, v in COMBO_PREFIX.items()}

# 描述性文档
KEY_DESCRIPTIONS: Dict[str, str] = {
    "Enter": "回车键",
    "Tab": "制表键（切换焦点）",
    "Escape": "退出键",
    "Backspace": "退格键（删除光标前字符）",
    "Delete": "删除键（删除光标后字符）",
    "Up/Down/Left/Right": "方向键",
    "F1-F12": "功能键",
    "Ctrl/Alt/Shift": "修饰键（支持组合键）",
    "Home/End": "跳转到行首/行尾",
    "PageUp/PageDown": "上页/下页",
    "Space": "空格键",
}


def is_combo_key(key_str: str) -> bool:
    """
    检查是否为组合键格式（包含组合键前缀）

    Args:
        key_str: 键字符串，如 '^c', '^+esc', '!F4'

    Returns:
        True 如果包含组合键前缀

    Example:
        >>> is_combo_key("^c")
        True
        >>> is_combo_key("Enter")
        False
    """
    if not key_str:
        return False
    return any(prefix in key_str for prefix in COMBO_PREFIX.keys())


def parse_combo_key(key_str: str) -> Tuple[List[str], str]:
    """
    解析组合键字符串，提取修饰键和主键

    Args:
        key_str: 组合键字符串，如 '^c', '^+esc', '!F4', '#e'

    Returns:
        (modifiers, main_key) 元组
        例如: '^c' -> (['Ctrl'], 'c')
              '^+esc' -> (['Ctrl', 'Shift'], 'esc')
              '!F4' -> (['Alt'], 'F4')

    Example:
        >>> parse_combo_key("^c")
        (['Ctrl'], 'c')
        >>> parse_combo_key("^+esc")
        (['Ctrl', 'Shift'], 'esc')
    """
    if not key_str:
        return ([], "")

    modifiers = []
    remaining = key_str

    # 按固定顺序提取前缀，确保多前缀正确解析
    for prefix, name in COMBO_PREFIX.items():
        if prefix in remaining:
            modifiers.append(name)
            remaining = remaining.replace(prefix, "", 1)

    return (modifiers, remaining)


def get_tcp_combo_code(modifiers: List[str], key: str) -> str:
    """
    生成组合键的 TCP 命令字符串

    Args:
        modifiers: 修饰键列表，如 ['Ctrl', 'Shift']
        key: 主键，如 'c', 'esc', 'F4'

    Returns:
        组合键字符串，如 '^+c', '!F4', '#e'

    Example:
        >>> get_tcp_combo_code(['Ctrl'], 'c')
        '^c'
        >>> get_tcp_combo_code(['Ctrl', 'Shift'], 'esc')
        '^+esc'
    """
    combo_str = ""
    for mod in modifiers:
        prefix = MODIFIER_TO_PREFIX.get(mod, mod)
        combo_str += prefix
    combo_str += key
    return combo_str


def get_tcp_key_code(key_name: str) -> str:
    """
    获取 TCP 命令格式的键码

    Args:
        key_name: 键名或组合键字符串

    Returns:
        普通键: "{Enter}", "{F5}" 等
        组合键: "^c", "^+esc" 等

    Raises:
        KeyError: 键名不在映射表中（仅对普通键）

    Example:
        >>> get_tcp_key_code("Enter")
        "{Enter}"
        >>> get_tcp_key_code("^c")
        "^c"
    """
    # 如果是组合键格式，直接解析并生成
    if is_combo_key(key_name):
        modifiers, main_key = parse_combo_key(key_name)
        return get_tcp_combo_code(modifiers, main_key)

    # 普通键查找映射表
    normalized = normalize_key_name(key_name)
    if normalized not in KEY_MAP:
        raise KeyError(f"Unknown key: {key_name}. "
                      f"Valid keys: {', '.join(sorted(VALID_SPECIAL_KEYS))}")
    return KEY_MAP[normalized]


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
        True 如果是特殊键（在 KEY_MAP 中）或组合键格式
    """
    if is_combo_key(key_name):
        return True
    return normalize_key_name(key_name) in VALID_SPECIAL_KEYS


if __name__ == "__main__":
    # 简单测试
    print("Key Map Tests:")
    print(f"  normalize_key_name('enter') = {normalize_key_name('enter')}")
    print(f"  normalize_key_name('ESC') = {normalize_key_name('ESC')}")
    print(f"  is_special_key('Enter') = {is_special_key('Enter')}")
    print(f"  is_special_key('InvalidKey') = {is_special_key('InvalidKey')}")
    print(f"  get_tcp_key_code('Enter') = {get_tcp_key_code('Enter')}")
    print(f"  get_tcp_key_code('F5') = {get_tcp_key_code('F5')}")
    print()
    print("Combo Key Tests:")
    print(f"  is_combo_key('^c') = {is_combo_key('^c')}")
    print(f"  is_combo_key('Enter') = {is_combo_key('Enter')}")
    print(f"  parse_combo_key('^c') = {parse_combo_key('^c')}")
    print(f"  parse_combo_key('^+esc') = {parse_combo_key('^+esc')}")
    print(f"  parse_combo_key('!F4') = {parse_combo_key('!F4')}")
    print(f"  get_tcp_key_code('^c') = {get_tcp_key_code('^c')}")
    print(f"  get_tcp_key_code('^+esc') = {get_tcp_key_code('^+esc')}")
    print(f"  get_tcp_key_code('!F4') = {get_tcp_key_code('!F4')}")
    print(f"  get_tcp_key_code('#e') = {get_tcp_key_code('#e')}")
