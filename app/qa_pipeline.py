"""
Question-answering pipeline (LOCAL MODEL - text-only RAG).

Steps:
- Build an index from a PDF (using previous steps).
- For a user query:
    - Retrieve top-k similar chunks
    - Format them as context
    - Run a local model (FLAN-T5) to generate an answer
"""

from typing import List, Dict, Any, Tuple

from transformers import pipeline

from app.ingestion import get_pdf_page_texts, extract_image_ocr_chunks, extract_table_chunks

from app.chunking import chunk_pdf_pages_to_document_chunks
from app.index import build_text_index, search_text_index
from app.models import DocumentChunk


# Load local model ONCE (this may take time the first time)
print("⏳ Loading local LLM: google/flan-t5-base (first time may be slow)...")
local_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)
print("✅ Local LLM loaded successfully!")


def build_qa_index_from_pdf(
    pdf_name: str = "qatar_test_doc.pdf",
    doc_id: str = "qatar_report",
) -> Dict[str, Any]:

    """
    Reads the PDF -> creates:
      - text chunks (from page text)
      - image_ocr chunks (from images via OCR)
    Then builds a single vector index over ALL chunks.
    """
    # 1) Page text → text chunks
    page_texts = get_pdf_page_texts(pdf_name)
    text_chunks = chunk_pdf_pages_to_document_chunks(doc_id, page_texts, max_words=200)

    # 2) Images → OCR chunks
    ocr_chunks = extract_image_ocr_chunks(pdf_name=pdf_name, doc_id=doc_id)

    # 3) Tables → table chunks
    table_chunks = extract_table_chunks(pdf_name=pdf_name, doc_id=doc_id)

    # 4) Combine all chunks
    all_chunks = text_chunks + ocr_chunks + table_chunks

    print(
        f"Text chunks: {len(text_chunks)}, "
        f"OCR chunks: {len(ocr_chunks)}, "
        f"Table chunks: {len(table_chunks)}, "
        f"Total: {len(all_chunks)}"
    )


    print(f"Text chunks: {len(text_chunks)}, OCR chunks: {len(ocr_chunks)}, total: {len(all_chunks)}")

    # 4) Build index over all of them
    index = build_text_index(all_chunks)
    return index



def format_context_for_prompt(chunks: List[DocumentChunk]) -> str:
    """
    Convert chunks into a text block for input to the LLM.
    We truncate each chunk to avoid exceeding model limits.
    """
    lines = []
    for i, ch in enumerate(chunks, start=1):
        header = f"[Chunk {i} | Page {ch.page} | Modality={ch.modality}]"
        # Truncate content to first 500 characters to keep prompt reasonable
        snippet = ch.content[:500]
        lines.append(header)
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)



def answer_question(
    index: Dict[str, Any],
    query: str,
    top_k: int = 5,
) -> Tuple[str, List[DocumentChunk]]:
    """
    Core QA function using LOCAL AI model
    """
    # 1. Retrieve top-k chunks
    results = search_text_index(index, query, top_k=top_k)
    retrieved_chunks = [chunk for score, chunk in results]

    if not retrieved_chunks:
        return "I couldn't find any relevant information in the document.", []

    # 2. Format context
    context_text = format_context_for_prompt(retrieved_chunks)

    # 3. Build final prompt for the LOCAL model
    final_prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{context_text}

Question:
{query}

If the answer is not in the context, say:
"I cannot find this information in the document."

Answer:
"""

    # 4. Run local model
    output = local_llm(final_prompt)

    answer_text = output[0]["generated_text"].strip()

    return answer_text, retrieved_chunks
