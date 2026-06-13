# phase6_flask_api.py
# Purpose: Wrap the RAG pipeline in a Flask REST API.
# Exposes a /ask endpoint that accepts a question and returns a grounded answer.

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from phase2_document_loader import load_pdf
from phase3_chunker import chunk_pages
from phase4_embeddings import build_vector_store, search
from phase5_rag_pipeline import ask

# --- Load environment variables ---
load_dotenv()

# --- Initialise Flask ---
# __name__ tells Flask where to find resources relative to this file
app = Flask(__name__)

# --- Load and index documents at startup ---
# We do this ONCE when the server starts — not on every request
# Loading 1,533 vectors on every question would be far too slow
print("Initialising Domain Assistant...")

filepath = os.path.join("documents", "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf")

print("Loading PDF...")
pages = load_pdf(filepath)

print("Chunking...")
chunks = chunk_pages(pages)

print("Building vector store...")
index, embedding_model = build_vector_store(chunks)

print("Domain Assistant ready. Server starting...\n")


# --- Define the /ask endpoint ---
@app.route("/ask", methods=["POST"])
def ask_question():
    """
    POST /ask
    Expects JSON body: {"question": "your question here"}
    Returns JSON: {"question": "...", "answer": "...", "sources": [...]}
    """

    # --- Get the request body ---
    # request.get_json() parses the incoming JSON into a Python dictionary
    data = request.get_json()

    # --- Validate the input ---
    # If no question was provided, return a 400 Bad Request error
    if not data or "question" not in data:
        return jsonify({"error": "Please provide a question in the request body"}), 400

    question = data["question"]

    # --- Guard against empty questions ---
    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    print(f"Received question: {question}")

    # --- Run the RAG pipeline ---
    # This is the same ask() function we built in Phase 5
    result = ask(question, index, embedding_model, chunks)

    # --- Return the result as JSON ---
    # jsonify() converts our Python dictionary into a proper JSON response
    return jsonify(result)


# --- Health check endpoint ---
# A simple GET endpoint that confirms the server is running
# Standard practice in any production API
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "documents_loaded": 1,
        "chunks_indexed": len(chunks)
    })


# --- Run the server ---
if __name__ == "__main__":
    # debug=False for stability
    # port=5000 is the Flask default
    # host="0.0.0.0" makes it accessible on your local network
    app.run(debug=False, port=5001, host="0.0.0.0")