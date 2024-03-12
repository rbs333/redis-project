import os
from redis import Redis
from dotenv import load_dotenv
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from db.doc_vector import DocVector, hf


recipe_schema: IndexSchema = IndexSchema.from_dict(
    {
        "index": {"name": "idx:recipes", "prefix": "recipe"},
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


def get_vector_index(client):
    return SearchIndex(recipe_schema, client)


def init_vector_store():
    print("Loading redis data \n")
    client = Redis.from_url(os.environ["REDIS_URL"])
    index = get_vector_index(client)
    index.create(overwrite=True, drop=True)

    # read files from data/docs - could be more fancy
    doc_path = os.path.join(os.path.abspath(os.curdir), "data/docs/")
    docs = [
        f"{doc_path}/{file}"
        for file in os.listdir(doc_path)
        if os.path.isfile(os.path.join(doc_path, file)) and "sample" in file
    ]

    print(f"Docs to be processed: {docs}\n\n")

    # create context and load into database
    system_prompt = """
      You are a recipe recommendation tool. You will be provided segments of a raw pdfs containing recipes to process.

      Create a clear proposition from the data which includes all potentially important details about the recipe. Make sure to add
      descriptors of the recipes such as "vegetarian", "hearty", or "easy" to make more searchable.

      Return the proposition as a single string with key proposition in a json like so: {"proposition": "single string"}
    """

    context = DocVector(docs, system_prompt)
    redis_data = context.get_redis_data()
    index.load(redis_data, id_field="chunk_id")

    print("data loaded!\n\n")


if __name__ == "__main__":
    load_dotenv()
    init_vector_store()
