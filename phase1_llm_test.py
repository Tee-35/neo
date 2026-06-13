# phase1_llm_test.py
# Purpose: Confirm that Python can connect to the OpenAI API and return a response.
# This is the foundation of the Domain Assistant — everything else feeds into this call.

import os                        # Built-in Python library for interacting with the operating system
from dotenv import load_dotenv   # Reads our .env file and loads the variables into Python's environment
from openai import OpenAI        # The official OpenAI Python SDK

# --- Load environment variables ---
# This reads the .env file and makes OPENAI_API_KEY available via os.getenv()
# Without this line, Python has no idea the .env file exists
load_dotenv()

# --- Initialise the OpenAI client ---
# The SDK looks for OPENAI_API_KEY in the environment automatically
# We never write the key directly in the code — it stays in .env
client = OpenAI()

# --- Define a test question ---
test_question = "What is RAG in the context of AI applications?"

# --- Make the API call ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": test_question}
    ]
)

# --- Extract and print the response ---
print("Making API call...")
answer = response.choices[0].message.content

print("Question:", test_question)
print("\nAnswer:", answer)
