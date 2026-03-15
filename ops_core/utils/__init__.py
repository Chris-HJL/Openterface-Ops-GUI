"""
工具模块包
提供键鼠操作相关的工具功能
"""

from .key_map import (
    KEY_MAP,
    VALID_SPECIAL_KEYS,
    normalize_key_name,
    is_special_key,
    get_tcp_key_code
)

from .text_splitter import (
    TextSplitter,
    MAX_CHARS_PER_COMMAND,
    DEFAULT_DELAY_BETWEEN_CHUNKS
)

from .command_sequence import (
    CommandItem,
    CommandSequence,
    COMMON_SEQUENCES,
    get_common_sequence
)

__all__ = [
    # key_map
    'KEY_MAP',
    'VALID_SPECIAL_KEYS',
    'normalize_key_name',
    'is_special_key',
    'get_tcp_key_code',
    
    # text_splitter
    'TextSplitter',
    'MAX_CHARS_PER_COMMAND',
    'DEFAULT_DELAY_BETWEEN_CHUNKS',
    
    # command_sequence
    'CommandItem',
    'CommandSequence',
    'COMMON_SEQUENCES',
    'get_common_sequence',
]
