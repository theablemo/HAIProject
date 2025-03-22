# Initiate flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os


app = Flask(__name__)
CORS(app)

# Load the Whisper model
model = whisper.load_model("base")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
