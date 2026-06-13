# phase3_chunker.py
# Purpose: Split extracted PDF pages into smaller overlapping chunks.
# Smaller chunks = more precise retrieval in Phase 4.

import os
from phase2_document_loader import load_pdf    # We import our function from Phase 2


def chunk_pages(pages, chunk_size=500, overlap=50):
    """
    Takes a list of pages (from load_pdf) and splits each page's text
    into overlapping chunks of roughly chunk_size characters.

    chunk_size: how many characters per chunk (500 is a good starting point)
    overlap: how many characters to repeat between chunks (prevents losing
             meaning at boundaries)

    Returns a list of chunk dictionaries.
    """

    chunks = []   # This will hold all our chunks across all pages

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_number = page["page_number"]

        # --- Split this page's text into chunks ---
        # We step through the text in increments of (chunk_size - overlap)
        # The overlap means each chunk shares some text with the next one
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Slice the text from start to end
            chunk_text = text[start:end]

            # Only keep chunks that have meaningful content
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page_number": page_number,
                    "chunk_index": len(chunks)   # A unique ID for this chunk
                })

            # Move forward by (chunk_size - overlap) so chunks overlap slightly
            start += chunk_size - overlap

    return chunks


# --- List comprehension in action ---
# Once we have all chunks, we can use a list comprehension to do quick analysis
# This builds a list of just the text lengths — one number per chunk
def summarise_chunks(chunks):
    """Prints a summary of the chunks we've created."""

    # List comprehension: build a list of lengths from our chunks
    # Read as: "the length of chunk's text, for each chunk in chunks"
    lengths = [len(chunk["text"]) for chunk in chunks]

    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(lengths) // len(lengths)} characters")
    print(f"Smallest chunk: {min(lengths)} characters")
    print(f"Largest chunk: {max(lengths)} characters")
    print(f"\n--- Sample chunk (chunk 10) ---\n")
    print(chunks[10]["text"])
    print(f"\nSource: {chunks[10]['source']}, Page: {chunks[10]['page_number']}")


# --- Test it ---
filepath = os.path.join("documents", "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf")

print("Loading PDF...")
pages = load_pdf(filepath)

print("Chunking pages...")
chunks = chunk_pages(pages)

summarise_chunks(chunks)

