class BaseVectorStore:
    """Abstract interface for vector database backends."""

    def create_index(self):
        raise NotImplementedError

    def insert(self, vectors, metadata=None):
        raise NotImplementedError

    def query(self, vector, k=5, filters=None):
        raise NotImplementedError

