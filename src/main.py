from embeddings.embedding_pipeline import EmbeddingPipeline
from vectorstores.pinecone_store import PineconeVectorStore
from rag.retriever import Retriever
from rag.context_builder import ContextBuilder
from rag.generator import Generator

OPENAI_KEY = "LA_TUA_API_KEY"
PINECONE_KEY = "LA_TUA_API_KEY"
PINECONE_ENV = "us-east1-gcp"

def main():
    embedding_model = EmbeddingPipeline(
        model_name="text-embedding-3-large",
        api_key=OPENAI_KEY
    )

    vector_store = PineconeVectorStore(
        index_name="rag-index",
        api_key=PINECONE_KEY,
        environment=PINECONE_ENV
    )

    retriever = Retriever(vector_store=vector_store, k=5)
    context_builder = ContextBuilder(max_tokens=1024)
    generator = Generator(model_name="gpt-4o-mini", api_key=OPENAI_KEY)

    # Inserimento documenti
    docs = [
        "Vector databases enable semantic search.",
        "RAG systems combine retrieval with generation."
    ]
    embeddings = embedding_model.embed_texts(docs)
    metadata = [{"text": d} for d in docs]
    vector_store.insert(embeddings, metadata)

    # Query
    query = "What is a vector database?"
    query_embedding = embedding_model.embed_texts([query])[0]
    results = retriever.retrieve(query_embedding)
    context = context_builder.build_context([m.metadata for m in results])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = generator.generate(prompt)

    print(answer)

if __name__ == "__main__":
    main()

