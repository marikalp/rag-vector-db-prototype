from typing import List, Dict, Any, Optional
from .base_vectorstore import BaseVectorStore
import pinecone

class PineconeVectorStore(BaseVectorStore):
    def __init__(self, index_name: str, api_key: str, environment: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=1536, metric="cosine")

        self.index = pinecone.Index(index_name)

    def create_index(self):
        pass  # già creato nel costruttore

    def insert(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        items = []
        for i, vec in enumerate(vectors):
            items.append({
                "id": f"doc-{i}",
                "values": vec,
                "metadata": metadata[i] if metadata else {}
            })
        self.index.upsert(items)

    def query(self, vector: List[float], k: int = 5, filters: Optional[Dict[str, Any]] = None):
        result = self.index.query(
            vector=vector,
            top_k=k,
            include_metadata=True,
            filter=filters
        )
        return result.matches
