from typing import List, Dict, Any
from ..vectorstores.base_vectorstore import BaseVectorStore

class Retriever:
    """
    Component responsible for retrieving relevant documents from a vector store.
    """

    def __init__(self, vector_store: BaseVectorStore, k: int = 5):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query_embedding: List[float], filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        return self.vector_store.query(query_embedding, k=self.k, filters=filters)

