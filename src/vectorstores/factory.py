from src.vectorstores.pinecone_store import PineconeVectorStore
from src.vectorstores.chroma_store import ChromaVectorStore
from src.vectorstores.milvus_store import MilvusVectorStore
from src.vectorstores.weaviate_store import WeaviateVectorStore

class VectorStoreFactory:
    @staticmethod
    def create(db_type: str, **kwargs):
        db_type = db_type.lower()

        if db_type == "pinecone":
            return PineconeVectorStore(
                index_name=kwargs.get("index_name", "rag-index"),
                api_key=kwargs.get("api_key"),
                cloud=kwargs.get("cloud", "aws"),
                region=kwargs.get("region", "us-east-1")
            )

        if db_type == "chroma":
            return ChromaVectorStore(
                index_name=kwargs.get("index_name", "rag-index")
            )

        if db_type == "milvus":
            return MilvusVectorStore(
                index_name=kwargs.get("index_name", "rag-index"),
                dim=kwargs.get("dim", 1536),
                uri="milvus.db"
            )

        if db_type == "weaviate":
            return WeaviateVectorStore(index_name="RAGDocument")

        raise ValueError(f"Database vettoriale non supportato: {db_type}")


