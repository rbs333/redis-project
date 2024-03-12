import os
import json
import redis
from dotenv import load_dotenv
from redis.commands.json.path import Path
from redis.commands.search.field import TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


user_schema = (
    TextField("$.username", as_name="username"),
    TextField("$.password", as_name="password"),
    TextField("$.userId", as_name="userId"),
)


def init_user_memory_index():
    client = redis.Redis.from_url(os.environ["REDIS_URL"])

    rs = client.ft("idx:users")

    rs.create_index(
        user_schema,
        definition=IndexDefinition(prefix=["user:"], index_type=IndexType.JSON),
    )

    with open(
        os.path.join(os.path.abspath(os.curdir), "../db/seed/users.json"), "r"
    ) as file:
        users = json.load(file)

    for i, user in enumerate(users):
        client.json().set(f"user:{i}", Path.root_path(), user)


if __name__ == "__main__":
    load_dotenv()
    init_user_memory_index()
