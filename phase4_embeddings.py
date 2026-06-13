# phase4_embeddings.py
# Purpose: Convert text chunks into vectors and store them in a searchable FAISS index.
# This is the retrieval layer of our RAG pipeline.

import os
import numpy as np                                    # NumPy handles the numerical arrays (vectors)
from sentence_transformers import SentenceTransformer # The embedding model
import faiss                                          # Facebook's vector search library

from phase2_document_loader import load_pdf
from phase3_chunker import chunk_pages


def build_vector_store(chunks):
    """
    Takes a list of chunks and:
    1. Converts each chunk's text into a vector using an embedding model
    2. Stores all vectors in a FAISS index for fast similarity search
    
    Returns the FAISS index and the embedding model (we need both later).
    """

    # --- Load the embedding model ---
    # all-MiniLM-L6-v2 is small, fast, and excellent for semantic search
    # It will download automatically on first run (~80MB)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Extract just the text from each chunk ---
    # List comprehension: build a list of strings from our chunk dictionaries
    texts = [chunk["text"] for chunk in chunks]

    # --- Convert all texts to vectors ---
    # model.encode() runs every chunk through the embedding model
    # show_progress_bar=True gives us a progress indicator — 1,533 chunks takes a moment
    print("Generating embeddings for all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # --- Convert to the format FAISS expects ---
    # FAISS needs a NumPy array of 32-bit floats
    embeddings = np.array(embeddings).astype("float32")

    # --- Build the FAISS index ---
    # embeddings.shape[1] is the vector dimension — 384 for this model
    # IndexFlatL2 measures straight-line distance between vectors (L2 = Euclidean distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # --- Add all our vectors to the index ---
    index.add(embeddings)
    print(f"Vector store built. {index.ntotal} vectors indexed.")

    return index, model


def search(query, index, model, chunks, top_k=5):
    """
    Takes a natural language question and returns the top_k most relevant chunks.
    
    query: the user's question as a plain string
    top_k: how many chunks to return (5 is a good default for RAG)
    """

    # --- Convert the query to a vector ---
    # Same model, same process as the chunks — so vectors are comparable
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # --- Search the index ---
    # index.search returns two arrays:
    # distances: how far each result is from the query (lower = more similar)
    # indices: the position of each result in our chunks list
    distances, indices = index.search(query_vector, top_k)

    # --- Build results ---
    # List comprehension: build a result list from the returned indices
    # We zip indices[0] and distances[0] to pair each index with its distance score
    results = [
        {
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "page_number": chunks[i]["page_number"],
            "score": float(distances[0][j])   # Lower score = more relevant
        }
        for j, i in enumerate(indices[0])
    ]

    return results


# --- Test it ---
filepath = os.path.join("documents", "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf")

print("Loading and chunking PDF...")
pages = load_pdf(filepath)
chunks = chunk_pages(pages)

# Build the vector store
index, model = build_vector_store(chunks)

# Test a search query
query = "What are the regulations for the front wing?"
print(f"\nSearching for: '{query}'\n")
results = search(query, index, model, chunks)

for i, result in enumerate(results):
    print(f"--- Result {i+1} (score: {result['score']:.2f}) ---")
    print(f"Source: {result['source']}, Page: {result['page_number']}")
    print(result["text"][:300])
    print()