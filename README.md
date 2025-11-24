# Multi-Modal RAG

This project implements a **multi-modal Retrieval-Augmented Generation (RAG)** system
that can answer questions about a PDF document (Qatar IMF-style report) using:

- Text extraction from PDF pages
- Image OCR (for figures / scanned content) using Tesseract
- Table extraction (where available)
- Embedding-based semantic search over all modalities
- A local LLM (google/flan-t5-base) for answer generation
- A Streamlit UI for interactive QA

---

## 1. Project Structure

```text
app/
  config.py
  models.py
  ingestion.py
  chunking.py
  embeddings.py
  index.py
  qa_pipeline.py

ui/
  streamlit_app.py

data/
  raw_docs/
    qatar_test_doc.pdf

test_ingestion.py
test_chunking.py
test_search.py
test_qa.py
eval_rag.py

requirements.txt
README.md


2. Setup

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

Make sure Tesseract is installed at:
C:\Program Files\Tesseract-OCR\tesseract.exe


3. Run the system (CLI)

python test_qa.py



4. Run the Streamlit UI
streamlit run ui/streamlit_app.py


5. Evaluation Script
python eval_rag.py
