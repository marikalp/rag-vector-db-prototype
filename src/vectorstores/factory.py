from src.vectorstores.pinecone_store import PineconeVectorStore
from src.vectorstores.chroma_store import ChromaVectorStore
from src.vectorstores.milvus_store import MilvusVectorStore
from src.vectorstores.weaviate_store import WeaviateVectorStore

class VectorStoreFactory:
    @staticmethod
    def create(db_type: str, **kwargs):
        db_type = db_type.lower()

        if db_type == "pinecone":
            return PineconeVectorStore(**kwargs)

        if db_type == "chroma":
            return ChromaVectorStore(**kwargs)

        if db_type == "milvus":
            return MilvusVectorStore(**kwargs)

        if db_type == "weaviate":
            return WeaviateVectorStore(**kwargs)

        raise ValueError(f"Database vettoriale non supportato: {db_type}")
