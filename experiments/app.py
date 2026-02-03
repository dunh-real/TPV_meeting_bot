from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello mother fucker"

if __name__ == "__main__":
    # Host '0.0.0.0' makes the server accessible
    app.run(host = '0.0.0.0', port = 5000)