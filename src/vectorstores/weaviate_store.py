import weaviate
from src.vectorstores.base_vectorstore import BaseVectorStore

class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, index_name="RAGDocument", url="http://localhost:9000"):
        self.index_name = index_name

        # Client v3 verso il proxy HTTP
        self.client = weaviate.Client(url)

        # Se la classe esiste già, la eliminiamo
        schema = self.client.schema.get()
        existing = [c["class"] for c in schema.get("classes", [])]
        if self.index_name in existing:
            self.client.schema.delete_class(self.index_name)

        # Creazione schema
        class_obj = {
            "class": self.index_name,
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]}
            ]
        }
        self.client.schema.create_class(class_obj)

    def insert(self, embeddings, metadata_list):
        with self.client.batch as batch:
            for emb, meta in zip(embeddings, metadata_list):
                batch.add_data_object(
                    data_object={"text": meta["text"]},
                    class_name=self.index_name,
                    vector=emb
                )

    def retrieve(self, query_embedding, k=5):
        result = (
            self.client.query
            .get(self.index_name, ["text"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(k)
            .do()
        )

        objs = result.get("data", {}).get("Get", {}).get(self.index_name, [])
        return [{"text": o["text"]} for o in objs]

    def clear(self):
        self.client.schema.delete_class(self.index_name)



