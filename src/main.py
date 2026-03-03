from src.config import OPENAI_API_KEY, PINECONE_API_KEY
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstores.factory import VectorStoreFactory
from src.rag.retriever import Retriever
from src.rag.context_builder import ContextBuilder
from src.rag.generator import Generator
from src.loader.pdf_loader import PDFLoader
from src.benchmark.benchmark import BenchmarkRunner

import random
import os


def run_interactive_rag():
    db_choice = input("Scegli il database (pinecone / chroma / milvus): ").strip()

    embedding_model = EmbeddingPipeline(
        model_name="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    vector_store = VectorStoreFactory.create(
        db_choice,
        index_name="rag-index",
        api_key=PINECONE_API_KEY,
        cloud="aws",
        region="us-east-1"
    )

    retriever = Retriever(vector_store=vector_store, k=5)
    context_builder = ContextBuilder(max_tokens=1024)
    generator = Generator(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)

    pdf_name = input("Inserisci il nome del PDF da caricare (es: documento.pdf): ").strip()
    pdf_path = f"pdfs/{pdf_name}"

    loader = PDFLoader(chunk_size=500, chunk_overlap=50)
    text = loader.load_pdf(pdf_path)
    chunks = loader.chunk_text(text)

    print(f"Caricati {len(chunks)} chunk dal PDF '{pdf_path}'.")

    embeddings = embedding_model.embed_texts(chunks)
    metadata = [{"text": chunk} for chunk in chunks]
    vector_store.insert(embeddings, metadata)

    print("\nIl documento è stato indicizzato. Ora puoi fare domande libere.")
    print("Scrivi 'exit' per uscire.\n")

    while True:
        query = input("Domanda: ").strip()
        if query.lower() == "exit":
            print("Uscita dal sistema.")
            break

        query_embedding = embedding_model.embed_texts([query])[0]
        results = retriever.retrieve(query_embedding)

        context_docs = []
        for r in results:
            if isinstance(r, dict) and "metadata" in r:
                context_docs.append(r["metadata"])
            elif isinstance(r, dict) and "text" in r:
                context_docs.append({"text": r["text"]})
            else:
                context_docs.append({"text": str(r)})

        context = context_builder.build_context(context_docs)

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = generator.generate(prompt)

        print("\n=== RISPOSTA ===\n")
        print(answer)
        print("\n-----------------------------\n")


def run_benchmark():
    print("Modalità benchmark attiva.")
    print("Userò un dataset sintetico di test.\n")

    # Dataset sintetico
    chunks = [f"Test chunk {i}" for i in range(200)]
    queries = [f"Query di test {i}" for i in range(20)]

    embedding_model = EmbeddingPipeline(
        model_name="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    embeddings = embedding_model.embed_texts(chunks)
    metadata = [{"text": c} for c in chunks]
    query_embeddings = embedding_model.embed_texts(queries)

    dbs = ["pinecone", "chroma", "milvus"]
    results = {}

    for db in dbs:
        print(f"\nBenchmarking {db.upper()}...")

        vector_store = VectorStoreFactory.create(
            db,
            index_name=f"benchmark-{db}",
            api_key=PINECONE_API_KEY,
            cloud="aws",
            region="us-east-1"
        )

        runner = BenchmarkRunner(
            vectorstore=vector_store,
            embeddings=embeddings,
            metadata=metadata,
            queries=query_embeddings,
            k=5
        )

        results[db] = runner.run()

    print("\n=== RISULTATI BENCHMARK ===\n")
    for db, r in results.items():
        print(f"\n--- {db.upper()} ---")
        print(f"Tempo indicizzazione: {r['index_time']:.4f}s")
        print(f"Throughput: {r['index_throughput']:.2f} vettori/s")
        print(f"Latenza media: {r['latency_mean']*1000:.2f} ms")
        print(f"Latenza p95: {r['latency_p95']*1000:.2f} ms")
        print(f"Latenza p99: {r['latency_p99']*1000:.2f} ms")


def main():
    mode = input("Scegli modalità (rag / benchmark): ").strip().lower()

    if mode == "rag":
        run_interactive_rag()
    elif mode == "benchmark":
        run_benchmark()
    else:
        print("Modalità non valida.")


if __name__ == "__main__":
    main()


