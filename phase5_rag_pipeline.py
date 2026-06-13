# phase5_rag_pipeline.py
# Purpose: Connect retrieval (Phase 4) to generation (Phase 1) to build the full RAG pipeline.
# This is the core of the Domain Assistant — it answers questions grounded in real documents.

import os
from dotenv import load_dotenv
from openai import OpenAI

from phase2_document_loader import load_pdf
from phase3_chunker import chunk_pages
from phase4_embeddings import build_vector_store, search

# --- Load environment variables ---
load_dotenv()
client = OpenAI()


def build_prompt(question, context_chunks):
    """
    Builds the prompt we send to the LLM.
    We give it the retrieved chunks as context and instruct it to
    answer only from that context — no hallucination.
    """

    # Build the context block from our retrieved chunks
    # Each chunk includes its text, source file, and page number
    context_block = ""
    for i, chunk in enumerate(context_chunks):
        context_block += f"[Source {i+1}: {chunk['source']}, Page {chunk['page_number']}]\n"
        context_block += chunk["text"]
        context_block += "\n\n"

    # The full prompt instructs the LLM clearly on how to behave
    prompt = f"""You are a technical expert on FIA Formula 1 regulations.
Your job is to answer questions accurately using ONLY the context provided below.
If the answer is not contained in the context, say "I could not find that information in the provided documents."
Always cite which source and page your answer comes from.

CONTEXT:
{context_block}

QUESTION: {question}

ANSWER:"""

    return prompt


def ask(question, index, model, chunks, top_k=5):
    """
    The main RAG function. Takes a question and returns a grounded answer.
    
    1. Retrieves the most relevant chunks using semantic search
    2. Builds a prompt with those chunks as context
    3. Sends the prompt to the LLM
    4. Returns the answer with source references
    """

    # --- Step 1: Retrieve relevant chunks ---
    print(f"Searching for relevant chunks...")
    relevant_chunks = search(question, index, model, chunks, top_k=top_k)

    # --- Step 2: Build the prompt ---
    prompt = build_prompt(question, relevant_chunks)

    # --- Step 3: Call the LLM ---
    print(f"Generating answer...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # --- Step 4: Extract the answer ---
    answer = response.choices[0].message.content

    # --- Step 5: Build the source list ---
    # List comprehension: build a deduplicated list of sources used
    sources = list({
        f"{chunk['source']} (Page {chunk['page_number']})"
        for chunk in relevant_chunks
    })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


# --- Test it ---
filepath = os.path.join("documents", "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf")

print("Loading and chunking PDF...")
pages = load_pdf(filepath)
chunks = chunk_pages(pages)

print("Building vector store...")
index, embedding_model = build_vector_store(chunks)

# --- Ask three test questions ---
questions = [
    "What are the regulations for the front wing dimensions?",
    "What materials are permitted for the survival cell?",
    "What are the rules around the power unit energy store?"
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)
    result = ask(question, index, embedding_model, chunks)
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result["sources"]:
        print(f"  - {source}")