from typing import List, Dict, Any, Optional

class BaseVectorStore:
    def create_index(self):
        raise NotImplementedError

    def insert(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]]):
        raise NotImplementedError

    def query(self, vector: List[float], k: int = 5, filters=None):
        raise NotImplementedError

