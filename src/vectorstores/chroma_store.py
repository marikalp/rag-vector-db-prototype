import chromadb
from chromadb.config import Settings
from src.vectorstores.base_vectorstore import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, index_name="rag-index"):
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet"))
        self.collection = self.client.get_or_create_collection(
            name=index_name,
            metadata={"hnsw:space": "cosine"}
        )

    def insert(self, embeddings, metadata_list):
        ids = [str(i) for i in range(len(embeddings))]
        texts = [m["text"] for m in metadata_list]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata_list
        )

    def retrieve(self, query_embedding, k=5):
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return res["metadatas"][0]

    def clear(self):
        self.client.delete_collection(self.collection.name)


