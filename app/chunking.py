"""
Functions for smartly splitting (chunking) long texts into smaller pieces.
STEP 4: Basic paragraph + word-limit chunking.
"""

from typing import List
import uuid

from app.models import DocumentChunk


def split_text_into_chunks(text: str, max_words: int = 200) -> List[str]:
    """
    Split a long text into smaller chunks:
    - First split by paragraphs (double newlines)
    - Then group paragraphs until we reach ~max_words
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    current = []
    current_word_count = 0

    for para in paragraphs:
        words_in_para = len(para.split())
        # If this paragraph alone is bigger than max_words, just cut it roughly
        if words_in_para > max_words:
            # If there is something in current, flush it first
            if current:
                chunks.append(" ".join(current))
                current = []
                current_word_count = 0

            words = para.split()
            for i in range(0, len(words), max_words):
                part = " ".join(words[i:i + max_words])
                chunks.append(part)
        else:
            # If adding this paragraph would exceed the limit, flush current
            if current_word_count + words_in_para > max_words and current:
                chunks.append(" ".join(current))
                current = [para]
                current_word_count = words_in_para
            else:
                current.append(para)
                current_word_count += words_in_para

    # Flush remaining
    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_pdf_pages_to_document_chunks(
    doc_id: str,
    page_texts: List[str],
    max_words: int = 200,
) -> List[DocumentChunk]:
    """
    Take a list of page texts and return a list of DocumentChunk objects.
    Each chunk knows which document and which page it came from.
    """

    all_chunks: List[DocumentChunk] = []

    for page_index, page_text in enumerate(page_texts):
        page_number = page_index + 1  # 1-based

        # Get text chunks for this page
        text_chunks = split_text_into_chunks(page_text, max_words=max_words)

        for chunk_text in text_chunks:
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                modality="text",
                page=page_number,
                content=chunk_text,
                extra=None,
            )
            all_chunks.append(chunk)

    return all_chunks
