import weaviate
from weaviate.classes.config import Configure, Property, DataType
from src.vectorstores.base_vectorstore import BaseVectorStore

class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, index_name="RAGDocument", url="http://localhost:8080"):
        self.index_name = index_name

        # Connessione corretta per client v4
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080
        )

        # Se la collection esiste già, la eliminiamo
        if self.index_name in self.client.collections.list_all():
            self.client.collections.delete(self.index_name)

        # Creazione collection
        self.collection = self.client.collections.create(
            name=self.index_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT)
            ]
        )

    def insert(self, embeddings, metadata_list):
        with self.collection.batch.dynamic() as batch:
            for emb, meta in zip(embeddings, metadata_list):
                batch.add_object(
                    properties={"text": meta["text"]},
                    vector=emb
                )

    def retrieve(self, query_embedding, k=5):
        results = (
            self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=k
            )
            .include_properties(["text"])
            .do()
        )

        output = []
        for obj in results.objects:
            output.append({
                "text": obj.properties["text"],
                "score": obj.distance
            })

        return output

    def clear(self):
        self.client.collections.delete(self.index_name)


