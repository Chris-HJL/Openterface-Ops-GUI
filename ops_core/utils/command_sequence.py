"""
命令序列工具
提供命令序列的数据结构、验证和优化功能
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class CommandItem:
    """单个命令项"""
    command: str
    delay: float = 0.5
    description: Optional[str] = None  # 可选的描述
    retry_count: int = 0  # 已重试次数
    max_retries: int = 3  # 最大重试次数

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "command": self.command,
            "delay": self.delay,
            "description": self.description,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CommandItem':
        """从字典创建"""
        return cls(
            command=data.get("command", ""),
            delay=data.get("delay", 0.5),
            description=data.get("description"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )

    def validate(self) -> Tuple[bool, str]:
        """
        验证命令项

        Returns:
            (is_valid, error_message)
        """
        if not self.command:
            return False, "Command cannot be empty"
        if self.delay < 0:
            return False, f"Delay cannot be negative: {self.delay}"
        if self.retry_count < 0:
            return False, f"Retry count cannot be negative: {self.retry_count}"
        if self.max_retries < 0:
            return False, f"Max retries cannot be negative: {self.max_retries}"
        return True, ""


@dataclass
class CommandSequence:
    """命令序列"""
    commands: List[CommandItem] = field(default_factory=list)
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def add_command(
        self,
        command: str,
        delay: float = 0.5,
        description: Optional[str] = None
    ) -> 'CommandSequence':
        """
        添加命令

        Args:
            command: 命令文本
            delay: 延迟（秒）
            description: 可选描述

        Returns:
            self（支持链式调用）
        """
        item = CommandItem(
            command=command,
            delay=delay,
            description=description
        )
        self.commands.append(item)
        return self

    def add_wait(self, duration: float, description: Optional[str] = None) -> 'CommandSequence':
        """
        添加等待

        Args:
            duration: 等待时长（秒）
            description: 可选描述

        Returns:
            self
        """
        return self.add_command("", delay=duration, description=description or f"Wait {duration}s")

    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证整个序列

        Returns:
            (is_valid, error_messages)
        """
        errors = []
        for i, item in enumerate(self.commands):
            is_valid, error = item.validate()
            if not is_valid:
                errors.append(f"Command {i}: {error}")
        return len(errors) == 0, errors

    def optimize(self) -> 'CommandSequence':
        """
        优化序列（合并连续的等待命令）

        Returns:
            优化后的序列（新实例）
        """
        if not self.commands:
            return self

        optimized = CommandSequence(
            name=self.name,
            description=self.description,
            metadata=self.metadata
        )

        i = 0
        while i < len(self.commands):
            item = self.commands[i]

            # 如果是等待命令，合并后续的等待
            if not item.command:
                total_delay = item.delay
                j = i + 1
                while j < len(self.commands) and not self.commands[j].command:
                    total_delay += self.commands[j].delay
                    j += 1

                # 添加合并后的等待命令
                optimized.add_command("", delay=total_delay, description=item.description)
                i = j
            else:
                # 普通命令直接添加
                optimized.commands.append(item)
                i += 1

        return optimized

    def to_dict_list(self) -> List[Dict]:
        """转换为字典列表（用于执行）"""
        return [cmd.to_dict() for cmd in self.commands]

    def to_json(self) -> str:
        """序列化为 JSON"""
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "commands": [cmd.to_dict() for cmd in self.commands]
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CommandSequence':
        """
        从 JSON 反序列化

        Args:
            json_str: JSON 字符串

        Returns:
            CommandSequence 实例

        Raises:
            json.JSONDecodeError: JSON 格式错误
            KeyError: 必需字段缺失
        """
        data = json.loads(json_str)
        sequence = cls(
            name=data.get("name"),
            description=data.get("description"),
            metadata=data.get("metadata", {})
        )
        sequence.commands = [
            CommandItem.from_dict(cmd) for cmd in data.get("commands", [])
        ]
        return sequence

    def __len__(self) -> int:
        """返回命令数量"""
        return len(self.commands)

    def __iter__(self):
        """迭代命令"""
        return iter(self.commands)

    def __repr__(self) -> str:
        """字符串表示"""
        name_str = f"name='{self.name}'" if self.name else ""
        return f"CommandSequence({len(self.commands)} commands, {name_str})"


# 预定义的常用序列
COMMON_SEQUENCES = {
    "refresh": CommandSequence(
        name="Refresh",
        description="刷新当前页面或窗口",
        commands=[CommandItem(command='Send "{F5}"', delay=2.0)]
    ),
    "screenshot": CommandSequence(
        name="Screenshot",
        description="截取屏幕",
        commands=[CommandItem(command='Send "{PrintScreen}"', delay=1.0)]
    ),
}


def get_common_sequence(name: str) -> Optional[CommandSequence]:
    """
    获取预定义的常用序列

    Args:
        name: 序列名称（refresh, screenshot 等）

    Returns:
        CommandSequence 实例或 None
    """
    return COMMON_SEQUENCES.get(name.lower())


if __name__ == "__main__":
    # 简单测试
    print("Command Sequence Tests:")
    
    # 创建序列
    seq = CommandSequence(name="Test Sequence", description="测试序列")
    seq.add_command('Send "Hello"', delay=0.5, description="输入文本")
    seq.add_command('Send "{Enter}"', delay=1.0, description="按回车")
    seq.add_wait(2.0, "等待响应")
    
    print(f"  Sequence: {seq}")
    print(f"  Commands: {len(seq)}")
    
    # 验证
    is_valid, errors = seq.validate()
    print(f"  Valid: {is_valid}, Errors: {errors}")
    
    # 优化
    optimized = seq.optimize()
    print(f"  Optimized: {optimized}")
    
    # 序列化
    json_str = seq.to_json()
    print(f"  JSON: {json_str[:100]}...")
    
    # 反序列化
    loaded = CommandSequence.from_json(json_str)
    print(f"  Loaded: {loaded}")
