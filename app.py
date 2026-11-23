import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, redirect, url_for, session, copy_current_request_context
from agent import ask_agent
from guardrails import check_message, parse_location, MAX_MESSAGE_LENGTH
from uuid import uuid4

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Refuse huge requests before they reach our code. Asking about the weather
# never needs more than this.
app.config["MAX_CONTENT_LENGTH"] = 10_000  # bytes

# The longest the assistant may think about one message, in seconds,
# before we give up and tell the user.
THINKING_TIME_LIMIT = 30

# Answers are worked out in the background, so we can stop waiting for them.
background = ThreadPoolExecutor(max_workers=4)

@app.route("/")
def home():
    if 'messages' not in session:
        session['messages'] = []

    return render_template("chat.html", messages=session.get('messages', []))

@app.route("/send", methods=["POST"])
def send():
    user_message = request.form.get("message", "")

    location = parse_location(request.form.get("latitude"), request.form.get("longitude"))
    if location:
        session["user_location"] = location

    thread_id = session.setdefault("thread_id", str(uuid4()))
    response = reply_in_time(user_message, thread_id)

    messages = session.setdefault('messages', [])
    messages.append({'type': 'human', 'content': user_message[:MAX_MESSAGE_LENGTH]})
    messages.append({'type': 'ai', 'content': response})
    session.modified = True
    return redirect(url_for("home"))

def reply(user_message, thread_id):
    """Check the message, and only ask the agent if it passes."""
    rejection = check_message(user_message)
    if rejection:
        # Rejected messages never reach the agent or its memory.
        return rejection
    return "".join(ask_agent(user_message, thread_id))

def reply_in_time(user_message, thread_id):
    """Like reply(), but gives up after THINKING_TIME_LIMIT seconds."""
    # The agent reads the user's location from the session, so the
    # background thread needs a copy of this request to see it.
    @copy_current_request_context
    def think():
        return reply(user_message, thread_id)

    try:
        return background.submit(think).result(timeout=THINKING_TIME_LIMIT)
    except TimeoutError:
        app.logger.warning("Gave up after %s seconds of thinking", THINKING_TIME_LIMIT)
        return "Sorry, that took too long. Please try again in a minute."
    except Exception:
        # Google's API is sometimes busy or down. Show a friendly message
        # instead of crashing the page, and keep the details in the log.
        app.logger.exception("Weather agent failed")
        return "Sorry, I'm having trouble right now. Please try again in a minute."

@app.route("/clear", methods=["POST"])
def clear():
    session['messages'] = []
    session['thread_id'] = str(uuid4())
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
