import os
from redis import Redis
from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit
from chat_app.chat import gen_answer
from db.llm_memory import LLMMemoryLayer
from db.vector_store import get_vector_index
from functools import wraps
import asyncio

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode=None)

# TODO: handle chat instance creation and management more elegantly.
client = Redis.from_url(os.environ["REDIS_URL"])
memory = LLMMemoryLayer(client)
vector_index = get_vector_index(client)


def async_action(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapped


@app.route("/")
def index():
    return render_template("index.html", async_mode=socketio.async_mode)


@socketio.event
@async_action
async def add_msg(data):
    # TODO: use something better like pydantic for deserialization
    user_id = data["userId"]
    msg = data["msg"]

    answer = await gen_answer(memory, user_id, msg, vector_index)
    emit("add_msg", {"username": "bot", "msg": answer})


if __name__ == "__main__":
    socketio.run(app)
