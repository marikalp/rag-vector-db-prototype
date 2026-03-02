from typing import List, Dict, Any, Optional
from .base_vectorstore import BaseVectorStore
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore(BaseVectorStore):
    def __init__(self, index_name: str, api_key: str, cloud: str = "aws", region: str = "us-east-1"):
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

        existing = self.pc.list_indexes().names()
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )

        self.index = self.pc.Index(index_name)

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
