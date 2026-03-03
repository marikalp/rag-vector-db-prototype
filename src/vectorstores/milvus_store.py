from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from src.vectorstores.base_vectorstore import BaseVectorStore

class MilvusVectorStore(BaseVectorStore):
    def __init__(self, index_name="rag_index", dim=1536, uri="milvus.db"):
        self.index_name = index_name
        self.dim = dim

        connections.connect("default", uri=uri)

        if utility.has_collection(index_name):
            utility.drop_collection(index_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]

        schema = CollectionSchema(fields)

        self.collection = Collection(
            name=index_name,
            schema=schema,
            using="default",
            shards_num=2
        )

        self.collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
        )

        self.collection.load()

    def insert(self, embeddings, metadata_list):
        texts = [m["text"] for m in metadata_list]
        self.collection.insert([embeddings, texts])

    def retrieve(self, query_embedding, k=5):
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["text"]
        )

        output = []
        for hit in results[0]:
            output.append({
                "text": hit.entity.get("text"),
                "score": hit.distance
            })

        return output

    def clear(self):
        utility.drop_collection(self.index_name)




