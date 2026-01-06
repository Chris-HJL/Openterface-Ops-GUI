"""
索引构建模块
"""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Dict, Any, Optional
from .readers import MHTMLReader
from config import Config

class IndexBuilder:
    """索引构建器类"""

    def __init__(self, api_base: Optional[str] = None, embed_model: Optional[str] = None):
        """
        初始化索引构建器

        Args:
            api_base: API基础URL
            embed_model: 嵌入模型名称
        """
        self.api_base = api_base or Config.RAG_API_BASE
        self.embed_model = embed_model or Config.RAG_EMBED_MODEL
        self._setup_llamaindex()

    def _setup_llamaindex(self):
        """配置LlamaIndex环境"""
        Settings.embed_model = OpenAIEmbedding(
            model_name=self.embed_model,
            api_base=self.api_base,
        )

    def build_index(
        self,
        docs_dir: str,
        index_dir: str,
        file_extractor: Optional[Dict[str, BaseReader]] = None
    ) -> bool:
        """
        从文档目录构建索引

        Args:
            docs_dir: 文档目录
            index_dir: 索引保存目录
            file_extractor: 文件提取器映射

        Returns:
            是否成功
        """
        try:
            from llama_index.core import SimpleDirectoryReader

            # 配置文件提取器
            if file_extractor is None:
                file_extractor = {
                    ".mhtml": MHTMLReader()
                }

            # 读取文档
            documents = SimpleDirectoryReader(
                docs_dir,
                file_extractor=file_extractor
            ).load_data()

            # 构建索引
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True,
            )

            # 保存索引
            index.storage_context.persist(persist_dir=index_dir)

            return True

        except Exception as e:
            print(f"Failed to build index: {str(e)}")
            return False
