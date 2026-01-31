"""
Index loading module
"""
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from typing import Optional
from config import Config

class IndexLoader:
    """Index loader class"""

    def __init__(self, api_base: Optional[str] = None, embed_model: Optional[str] = None):
        """
        Initialize index loader

        Args:
            api_base: API base URL
            embed_model: Embedding model name
        """
        self.api_base = api_base or Config.RAG_API_BASE
        self.embed_model = embed_model or Config.RAG_EMBED_MODEL
        self._setup_llamaindex()

    def _setup_llamaindex(self):
        """Configure LlamaIndex environment"""
        Settings.embed_model = OpenAIEmbedding(
            model_name=self.embed_model,
            api_base=self.api_base,
        )

    def load_index(self, index_dir: str) -> VectorStoreIndex:
        """
        Load index from directory

        Args:
            index_dir: Index directory

        Returns:
            Loaded index object
        """
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage(storage_context)
