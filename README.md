---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/neo.git
cd neo
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**
```bash
cp .env.example .env
# Edit .env and add your key
```

**5. Add your PDF documents to the `/documents` folder**

**6. Run the Streamlit UI**
```bash
streamlit run neo_app.py
```

Or run the Flask API:
```bash
python phase6_flask_api.py
```

---

## API reference

### POST /ask
```json
Request:  { "question": "What are the front wing dimension limits?" }
Response: { "question": "...", "answer": "...", "sources": [...] }
```

### GET /health
```json
{ "status": "healthy", "documents_loaded": 1, "chunks_indexed": 1533 }
```

---

## Example questions (FIA Technical Regulations demo)

- What are the front wing dimension limits?
- What are the power unit energy store rules?
- What is the maximum fuel flow rate?
- What are the crash structure requirements?
- What materials are permitted for the survival cell?

---

## Planned extensions

- Multi-document upload via the UI
- Support for `.txt` and `.docx` file formats
- Persistent vector store (save/reload without re-embedding)
- Confidence scoring on retrieved chunks
- Deployment to cloud (Streamlit Cloud or AWS)

---

## About

Built as part of a career transition into AI engineering. Demonstrates end-to-end RAG pipeline construction, LLM API integration, semantic search, REST API design, and practical AI tooling in Python.

**Tyrelle Newton** — Data Analyst → AI Engineer
[LinkedIn](#) · [GitHub](#)