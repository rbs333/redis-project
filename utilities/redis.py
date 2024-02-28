import settings
from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

hf = HFTextVectorizer(settings.VECTORIZER)


def define_schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {"name": settings.REDIS_INDEX_NAME, "prefix": "chunk"},
            "fields": [
                {"name": "chunk_id", "type": "tag", "attrs": {"sortable": True}},
                {"name": "raw_content", "type": "text"},
                {"name": "proposition", "type": "text"},
                {
                    "name": "text_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": hf.dims,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )
