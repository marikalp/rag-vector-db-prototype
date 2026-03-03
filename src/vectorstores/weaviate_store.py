import weaviate
from weaviate.classes.config import Configure, Property, DataType
from src.vectorstores.base_vectorstore import BaseVectorStore

class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, index_name="RAGDocument", url="http://localhost:9000"):
        self.index_name = index_name

        # Client v3-style verso il proxy locale
        self.client = weaviate.Client(url)

        # Se la collection esiste già, la eliminiamo
        schema = self.client.schema.get()
        existing_classes = [c["class"] for c in schema.get("classes", [])]
        if self.index_name in existing_classes:
            self.client.schema.delete_class(self.index_name)

        # Creazione schema
        class_obj = {
            "class": self.index_name,
            "vectorizer": "none",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"]
                }
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
        output = []
        for obj in objs:
            output.append({
                "text": obj.get("text", ""),
                # v3 non restituisce sempre la distanza qui, la omettiamo
            })
        return output

    def clear(self):
        self.client.schema.delete_class(self.index_name)

