"""
文档读取器模块
"""
import re
import quopri
from bs4 import BeautifulSoup
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import Dict, Any

class MHTMLReader(BaseReader):
    """自定义MHTML文件读取器"""

    def load_data(self, file_path: str, extra_info: dict = None) -> list[Document]:
        """
        加载MHTML文件并提取可见内容

        Args:
            file_path: MHTML文件路径
            extra_info: 额外信息

        Returns:
            文档列表
        """
        with open(file_path, 'rb') as f:
            mhtml_content = f.read()

        # 解码为字符串
        mhtml_content = mhtml_content.decode('utf-8', errors='ignore')

        # 查找HTML部分
        boundary_match = re.search(r'boundary="([^"]+)"', mhtml_content)
        if not boundary_match:
            return [Document(text="", extra_info=extra_info or {})]

        boundary = boundary_match.group(1)
        parts = mhtml_content.split(f'--{boundary}')

        html_content = ""
        for part in parts:
            if 'Content-Type: text/html' in part:
                part_content = re.sub(
                    r'Content-Type: text/html[\s\S]*?Content-Location: [^\n]*\n',
                    '',
                    part
                )
                part_content = part_content.strip()
                if part_content:
                    html_content = part_content
                    break

        if not html_content:
            return [Document(text="", extra_info=extra_info or {})]

        # 解码quoted-printable编码
        html_content = quopri.decodestring(html_content).decode('utf-8', errors='ignore')
        html_content = html_content.replace('\r\n', '\n')

        # 提取可见内容
        soup = BeautifulSoup(html_content, 'lxml')
        for script in soup(['script', 'style', 'noscript', 'meta', 'link', 'head']):
            script.extract()

        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n+', '\n', text)

        return [Document(text=text, extra_info=extra_info or {})]
