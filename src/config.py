import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY non trovata. Impostala nel terminale con: export OPENAI_API_KEY=...")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY non trovata. Impostala nel terminale con: export PINECONE_API_KEY=...")
