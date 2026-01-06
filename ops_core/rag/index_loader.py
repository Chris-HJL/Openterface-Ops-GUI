"""
索引加载模块
"""
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from typing import Optional
from config import Config

class IndexLoader:
    """索引加载器类"""

    def __init__(self, api_base: Optional[str] = None, embed_model: Optional[str] = None):
        """
        初始化索引加载器

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

    def load_index(self, index_dir: str) -> VectorStoreIndex:
        """
        从目录加载索引

        Args:
            index_dir: 索引目录

        Returns:
            加载的索引对象
        """
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage(storage_context)
