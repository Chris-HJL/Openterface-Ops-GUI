"""
Document retrieval module
"""
from typing import List, Optional
from llama_index.core import VectorStoreIndex
from .index_loader import IndexLoader

class DocumentRetriever:
    """Document retriever class"""

    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize document retriever

        Args:
            index_dir: Index directory
        """
        self.index_dir = index_dir
        self.index: Optional[VectorStoreIndex] = None
        self.retriever = None
        self.top_k = 3

    def _load_index_if_needed(self):
        """Load index if needed"""
        if self.index is None and self.index_dir:
            loader = IndexLoader()
            self.index = loader.load_index(self.index_dir)
            self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant documents based on query

        Args:
            query: Query text
            top_k: Number of documents to return

        Returns:
            List of retrieved document content
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
