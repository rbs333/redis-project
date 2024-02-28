from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import openai
import os
from dotenv import load_dotenv
import settings
from at_tool.context import Context
from at_tool.chat import Chat
from utilities.redis import define_schema
from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
import asyncio


# goal: build a RAG that answers questions about the AT
def init_db(client, index):
    # define docs to be loaded into context
    cwd = os.getcwd()
    docs = [os.path.join(cwd, "data/docs/white_blaze_sample_quick_info.pdf")]

    index.create(overwrite=True, drop=True)

    # create context and load into database
    context = Context(docs)
    redis_data = context.get_redis_data()
    print(f"Loading {redis_data=}")
    index.load(redis_data, id_field="chunk_id")


async def start_chat(chat):
    # move these to settings
    stopterms = ["exit", "quit", "end", "cancel"]

    # Simple Chat
    while True:
        most_recent_question = input()
        if most_recent_question.lower() in stopterms:
            break

        answer = await chat.answer_question(most_recent_question)
        print(answer, flush=True)


async def async_main(index):
    # next steps query and build chat function

    chat = Chat(index)  # init chat

    await start_chat(chat)


def main():
    load_dotenv()

    # init redis client
    client = Redis.from_url(os.environ["REDIS_URL"])

    # get schema definition
    schema = define_schema()

    # create an index from schema and the client
    index = SearchIndex(schema, client)

    indexes = client.execute_command("FT._LIST")

    if (
        f"b'{settings.REDIS_INDEX_NAME}'" not in [str(i) for i in indexes]
        or not client.keys()
    ):
        print("Index DNE or data has not been loaded => loading now!")
        init_db(client, index)

    print("Hello! Start your AT chat \n")
    asyncio.run(async_main(index))


if __name__ == "__main__":
    main()
