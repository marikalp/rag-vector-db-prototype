from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

class MilvusVectorStore(BaseVectorStore):
    def __init__(self, index_name="rag_index", dim=1536, uri="milvus.db"):
        self.index_name = index_name
        self.dim = dim

        connections.connect("default", uri=uri)

        # Se la collection esiste già, la eliminiamo per evitare conflitti
        if utility.has_collection(index_name):
            utility.drop_collection(index_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)
        ]

        schema = CollectionSchema(fields)
        self.collection = Collection(name=index_name, schema=schema)

        self.collection.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
        )

        self.collection.load()



