import chromadb
from chromadb.config import Settings
from src.vectorstores.base_vectorstore import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, index_name="rag-index", persist_directory="chroma_db"):
        self.index_name = index_name
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=index_name,
            metadata={"hnsw:space": "cosine"}
        )

    def insert(self, embeddings, metadata_list):
        ids = [f"doc_{i}" for i in range(len(embeddings))]
        documents = [m["text"] for m in metadata_list]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata_list
        )

    def retrieve(self, query_embedding, k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    def clear(self):
        self.client.delete_collection(self.index_name)


