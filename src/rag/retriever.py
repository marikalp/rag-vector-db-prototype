from typing import List, Dict, Any
from src.vectorstores.base_vectorstore import BaseVectorStore

class Retriever:
    def __init__(self, vector_store, k=5):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query_embedding, filters=None):
        return self.vector_store.retrieve(query_embedding, k=self.k)
