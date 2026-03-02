from src.config import OPENAI_API_KEY, PINECONE_API_KEY
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstores.pinecone_store import PineconeVectorStore
from src.rag.retriever import Retriever
from src.rag.context_builder import ContextBuilder
from src.rag.generator import Generator

def main():
    embedding_model = EmbeddingPipeline(
       model_name="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    vector_store = PineconeVectorStore(
        index_name="rag-index",
        api_key=PINECONE_API_KEY,
        cloud="aws",
        region="us-east-1"
    )

    retriever = Retriever(vector_store=vector_store, k=5)
    context_builder = ContextBuilder(max_tokens=1024)
    generator = Generator(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)

    docs = [
        "Vector databases enable semantic search.",
        "RAG systems combine retrieval with generation."
    ]
    embeddings = embedding_model.embed_texts(docs)
    metadata = [{"text": d} for d in docs]
    vector_store.insert(embeddings, metadata)

    query = "What is a vector database?"
    query_embedding = embedding_model.embed_texts([query])[0]
    results = retriever.retrieve(query_embedding)

    context_docs = [m.metadata for m in results]
    context = context_builder.build_context(context_docs)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = generator.generate(prompt)

    print("\n=== RAG ANSWER ===\n")
    print(answer)

if __name__ == "__main__":
    main()
