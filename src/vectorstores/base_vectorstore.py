from typing import List, Dict, Any, Optional

class BaseVectorStore:
    """Abstract interface for vector database backends."""

    def create_index(self) -> None:
        raise NotImplementedError

    def insert(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        raise NotImplementedError

    def query(self, vector: List[float], k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError
