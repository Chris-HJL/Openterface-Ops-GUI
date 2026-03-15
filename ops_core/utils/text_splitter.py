"""
文本拆分工具
处理长文本的自动拆分（respects 字符边界和语义）
"""

import re
from typing import List, Dict


# 配置常量
MAX_CHARS_PER_COMMAND = 25
DEFAULT_DELAY_BETWEEN_CHUNKS = 0.3


class TextSplitter:
    """智能文本拆分器"""

    def __init__(
        self,
        max_length: int = MAX_CHARS_PER_COMMAND,
        delay_between_chunks: float = DEFAULT_DELAY_BETWEEN_CHUNKS
    ):
        """
        初始化拆分器

        Args:
            max_length: 每块最大字符数（默认 25）
            delay_between_chunks: 块间延迟（秒）
        """
        self.max_length = max_length
        self.delay_between_chunks = delay_between_chunks

    def split(self, text: str) -> List[str]:
        """
        简单拆分（仅按字符数）

        Args:
            text: 原始文本

        Returns:
            拆分后的文本块列表

        Example:
            >>> splitter = TextSplitter(max_length=10)
            >>> splitter.split("01234567890123456789")
            ["0123456789", "0123456789"]
        """
        return self._simple_split(text)

    def split_preserving_words(self, text: str) -> List[str]:
        """
        拆分为命令（尽可能保持单词完整）

        Args:
            text: 原始文本

        Returns:
            拆分后的文本块列表

        Example:
            >>> splitter = TextSplitter(max_length=20)
            >>> splitter.split_preserving_words("Hello world, this is a test")
            ["Hello world, this", "is a test"]
        """
        if len(text) <= self.max_length:
            return [text]

        chunks = []
        i = 0

        while i < len(text):
            if i + self.max_length >= len(text):
                # 最后一块
                chunks.append(text[i:])
                break

            chunk = text[i:i + self.max_length]

            # 尝试在单词边界截断
            space_pos = chunk.rfind(' ')
            if space_pos != -1 and space_pos > self.max_length // 2:
                # 在空格处截断
                chunks.append(text[i:i + space_pos])
                i += space_pos
                # 跳过空格
                while i < len(text) and text[i] == ' ':
                    i += 1
            else:
                # 没有合适的位置，强制截断
                chunks.append(chunk)
                i += self.max_length

        return chunks

    def split_with_punctuation(self, text: str) -> List[str]:
        """
        拆分为命令（优先在标点处截断）

        Args:
            text: 原始文本

        Returns:
            拆分后的文本块列表

        Example:
            >>> splitter = TextSplitter(max_length=15)
            >>> splitter.split_with_punctuation("Hello. How are you? I'm fine.")
            ["Hello.", "How are you?", "I'm fine."]
        """
        if len(text) <= self.max_length:
            return [text]

        chunks = []
        i = 0

        while i < len(text):
            if i + self.max_length >= len(text):
                chunks.append(text[i:])
                break

            chunk = text[i:i + self.max_length]

            # 查找标点
            last_punct = -1
            for punct in ['.', ',', ';', ':', '!', '?', '\n']:
                pos = chunk.rfind(punct)
                if pos > self.max_length // 2:
                    last_punct = max(last_punct, pos)

            if last_punct != -1:
                # 在标点后截断（包含标点）
                chunks.append(text[i:i + last_punct + 1])
                i += last_punct + 1
                # 跳过空白
                while i < len(text) and text[i] in ' \t\n\r':
                    i += 1
            else:
                # 没有标点，强制截断
                chunks.append(chunk)
                i += self.max_length

        return chunks

    def split_by_sentences(self, text: str) -> List[str]:
        """
        按句子拆分（然后合并长句）

        Args:
            text: 原始文本

        Returns:
            拆分后的文本块列表（每块≤max_length）

        Example:
            >>> splitter = TextSplitter(max_length=25)
            >>> splitter.split_by_sentences("Hello. How are you? I'm doing well, thank you!")
            ["Hello.", "How are you?", "I'm doing well, thank"]
        """
        # 首先按句子拆分
        sentences = re.split(r'([.!?]+)', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 合并标点
        merged = []
        i = 0
        while i < len(sentences):
            if sentences[i].isalnum():  # 文本
                next_i = i + 1
                while next_i < len(sentences) and not sentences[next_i].isalnum():
                    # 合并标点
                    next_i += 1
                merged.append(''.join(sentences[i:next_i]))
                i = next_i
            else:
                i += 1

        # 确保每块不超过 max_length
        chunks = []
        for chunk in merged:
            if len(chunk) <= self.max_length:
                chunks.append(chunk)
            else:
                # 子拆分
                sub_chunks = self.split_preserving_words(chunk)
                chunks.extend(sub_chunks)

        return chunks

    def to_commands(self, text: str, strategy: str = 'simple') -> List[Dict]:
        """
        转换为命令序列

        Args:
            text: 原始文本
            strategy: 拆分策略 ('simple' | 'words' | 'punctuation' | 'sentences')

        Returns:
            [
                {"command": 'Send "chunk1"', "delay": 0.3},
                {"command": 'Send "chunk2"', "delay": 0.3},
                ...
            ]

        Example:
            >>> splitter = TextSplitter()
            >>> commands = splitter.to_commands("Hello world. How are you?")
            >>> commands
            [{'command': 'Send "Hello world."', 'delay': 0.3}, ...]
        """
        strategy_map = {
            'simple': self.split,
            'words': self.split_preserving_words,
            'punctuation': self.split_with_punctuation,
            'sentences': self.split_by_sentences
        }

        split_method = strategy_map.get(strategy, self.split)
        chunks = split_method(text)

        return [
            {"command": f'Send "{chunk}"', "delay": self.delay_between_chunks}
            for chunk in chunks
        ]

    def _simple_split(self, text: str) -> List[str]:
        """简单拆分（仅按字符数）"""
        if len(text) <= self.max_length:
            return [text]

        return [
            text[i:i + self.max_length]
            for i in range(0, len(text), self.max_length)
        ]


if __name__ == "__main__":
    # 简单测试
    print("Text Splitter Tests:")
    splitter = TextSplitter(max_length=15)
    
    text = "Hello world. How are you? I'm fine, thank you!"
    print(f"  Original: {text}")
    print(f"  Simple: {splitter.split(text)}")
    print(f"  Words: {splitter.split_preserving_words(text)}")
    print(f"  Punctuation: {splitter.split_with_punctuation(text)}")
    print(f"  Sentences: {splitter.split_by_sentences(text)}")
    
    # 测试命令转换
    commands = splitter.to_commands(text, 'simple')
    print(f"  Commands: {commands}")
