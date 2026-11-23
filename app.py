from flask import Flask, render_template, request, redirect, url_for, session
from agent import ask_agent
from uuid import uuid4

app = Flask(__name__)
app.secret_key = 'mahsa-dev'

@app.route("/")
def home():
    if 'messages' not in session:
        session['messages'] = []

    return render_template("chat.html", messages=session.get('messages', []))

@app.route("/send", methods=["POST"])
def send():
    user_message = request.form["message"]
    user_lat = request.form.get("lat")
    user_lon = request.form.get("lon")

    thread_id = session.setdefault("thread_id", str(uuid4()))
    response = "".join(ask_agent(user_message, thread_id))

    session['messages'].append({'type': 'human', 'content': user_message})
    session['messages'].append({'type': 'ai', 'content': response})
    session.modified = True
    return redirect(url_for("home"))

@app.route("/clear", methods=["POST"])
def clear():
    session['messages'] = []
    session['thread_id'] = str(uuid4())
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)