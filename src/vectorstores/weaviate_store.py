import weaviate
from src.vectorstores.base_vectorstore import BaseVectorStore

class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, index_name="RAGDocument", url="http://localhost:8080"):
        self.index_name = index_name
        self.client = weaviate.Client(url)

        if not self.client.schema.exists(index_name):
            schema = {
                "classes": [{
                    "class": index_name,
                    "vectorizer": "none",
                    "properties": [
                        {"name": "text", "dataType": ["text"]}
                    ]
                }]
            }
            self.client.schema.create(schema)

    def insert(self, embeddings, metadata_list):
        for emb, meta in zip(embeddings, metadata_list):
            self.client.data_object.create(
                data_object={"text": meta["text"]},
                class_name=self.index_name,
                vector=emb
            )

    def retrieve(self, query_embedding, k=5):
        results = (
            self.client.query
            .get(self.index_name, ["text"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(k)
            .do()
        )
        return results["data"]["Get"][self.index_name]

    def clear(self):
        self.client.schema.delete_class(self.index_name)


