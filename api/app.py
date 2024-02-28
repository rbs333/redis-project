from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("index.html", async_mode=socketio.async_mode)


@socketio.event
def test_connection(data):
    print(f"pinged from browser {data}")
    emit("test_confirm", "confirmed test!")


@socketio.event
def add_msg(data):
    print(f"you asked bot: {data}")
    res = {"user": "bot", "msg": "cool story bro"}
    emit("add_msg", res)


@socketio.on("message")
def handle_message(data):
    print("received message: " + data)
    send("hello from the other side")


if __name__ == "__main__":
    socketio.run(app)
