"""
文档检索模块
"""
from typing import List, Optional
from llama_index.core import VectorStoreIndex
from .index_loader import IndexLoader

class DocumentRetriever:
    """文档检索器类"""

    def __init__(self, index_dir: Optional[str] = None):
        """
        初始化文档检索器

        Args:
            index_dir: 索引目录
        """
        self.index_dir = index_dir
        self.index: Optional[VectorStoreIndex] = None
        self.retriever = None
        self.top_k = 3

    def _load_index_if_needed(self):
        """如果需要，加载索引"""
        if self.index is None and self.index_dir:
            loader = IndexLoader()
            self.index = loader.load_index(self.index_dir)
            self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        根据查询检索相关文档

        Args:
            query: 查询文本
            top_k: 返回的文档数量

        Returns:
            检索到的文档内容列表
        """
        if top_k:
            self.top_k = top_k

        self._load_index_if_needed()

        if not self.retriever:
            return []

        try:
            query_results = self.retriever.retrieve(query)

            retrieved_content = []
            for result in query_results:
                node = result.node
                content = node.text.strip()
                retrieved_content.append(content)

            return retrieved_content

        except Exception as e:
            print(f"Failed to retrieve documents: {str(e)}")
            return []
