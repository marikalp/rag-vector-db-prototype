import os
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from src.vectorstores.base_vectorstore import BaseVectorStore

class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, index_name="RAGDocument", url=None):
        self.index_name = index_name

        # Legge URL e API key dalle variabili d'ambiente
        cluster_url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")

        if cluster_url is None or api_key is None:
            raise ValueError("WEAVIATE_URL e WEAVIATE_API_KEY devono essere impostate nel terminale.")

        # Connessione al cluster Weaviate Cloud
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=weaviate.AuthApiKey(api_key)
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



