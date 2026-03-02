from src.config import OPENAI_API_KEY, PINECONE_API_KEY
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstores.pinecone_store import PineconeVectorStore
from src.rag.retriever import Retriever
from src.rag.context_builder import ContextBuilder
from src.rag.generator import Generator
from src.loader.pdf_loader import PDFLoader

def main():
    # MODELLI
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

        # CHIEDI IL PDF ALL'UTENTE
    pdf_path = input("Inserisci il nome del PDF da caricare (es: documento.pdf): ").strip()

    loader = PDFLoader(chunk_size=500, chunk_overlap=50)
    text = loader.load_pdf(pdf_path)
    chunks = loader.chunk_text(text)

    print(f"Caricati {len(chunks)} chunk dal PDF '{pdf_path}'.")


    print(f"Caricati {len(chunks)} chunk dal PDF.")

    # EMBEDDINGS + UPSERT
    embeddings = embedding_model.embed_texts(chunks)
    metadata = [{"text": chunk} for chunk in chunks]
    vector_store.insert(embeddings, metadata)

    # QUERY
    query = "Riassumi il contenuto del documento."
    query_embedding = embedding_model.embed_texts([query])[0]
    results = retriever.retrieve(query_embedding)

    context_docs = [m.metadata for m in results]
    context = context_builder.build_context(context_docs)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = generator.generate(prompt)

    print("\n=== RISPOSTA RAG ===\n")
    print(answer)

if __name__ == "__main__":
    main()
