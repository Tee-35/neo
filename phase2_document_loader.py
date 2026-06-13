# phase2_document_loader.py
# Purpose: Load a PDF file and extract its text content page by page.
# This becomes the document ingestion layer of our RAG pipeline.

import os                        # For building file paths that work on any operating system
from pypdf import PdfReader      # The library that knows how to read PDF files

def load_pdf(filepath):
    """
    Opens a PDF file and extracts all text from every page.
    Returns a list of dictionaries — one per page — containing
    the page number and the extracted text.
    """

    # --- Open the PDF ---
    # PdfReader handles the binary PDF format for us
    # We pass it the filepath and it gives us back a reader object
    reader = PdfReader(filepath)

    # --- Extract text page by page ---
    # reader.pages is a list of page objects
    # We loop through each one and pull out the text
    pages = []

    for i, page in enumerate(reader.pages):

        # page.extract_text() returns the text content of that page as a string
        text = page.extract_text()

        # Some pages (diagrams, images) may return None — we skip those
        if text and text.strip():
            pages.append({
                "page_number": i + 1,      # Human-readable page number (starts at 1)
                "text": text.strip(),       # .strip() removes leading/trailing whitespace
                "source": os.path.basename(filepath)   # Just the filename, not the full path
            })

    return pages


# --- Test it ---
# os.path.join builds a file path correctly regardless of operating system
# On Mac: documents/FIA_Technical.pdf
# On Windows: documents\FIA_Technical.pdf
filepath = os.path.join("documents", "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf")

print(f"Loading: {filepath}")
pages = load_pdf(filepath)

print(f"Pages extracted: {len(pages)}")
print(f"\n--- Page 1 preview ---\n")
print(pages[0]["text"][:500])   # Print first 500 characters of page 1 as a preview