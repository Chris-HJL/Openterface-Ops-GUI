"""
Index building module
"""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Dict, Any, Optional
from .readers import MHTMLReader
from config import Config

class IndexBuilder:
    """Index builder class"""

    def __init__(self, api_base: Optional[str] = None, embed_model: Optional[str] = None):
        """
        Initialize index builder

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

    def build_index(
        self,
        docs_dir: str,
        index_dir: str,
        file_extractor: Optional[Dict[str, BaseReader]] = None
    ) -> bool:
        """
        Build index from document directory

        Args:
            docs_dir: Documents directory
            index_dir: Index storage directory
            file_extractor: File extractor mapping

        Returns:
            Whether successful
        """
        try:
            from llama_index.core import SimpleDirectoryReader

            # Config file extractor
            if file_extractor is None:
                file_extractor = {
                    ".mhtml": MHTMLReader()
                }

            # Read documents
            documents = SimpleDirectoryReader(
                docs_dir,
                file_extractor=file_extractor
            ).load_data()

            # Build index
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True,
            )

            # Save index
            index.storage_context.persist(persist_dir=index_dir)

            return True

        except Exception as e:
            print(f"Failed to build index: {str(e)}")
            return False
