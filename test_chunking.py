from app.ingestion import get_pdf_page_texts
from app.chunking import chunk_pdf_pages_to_document_chunks


def main():
    # 1) Load page texts from the sample PDF
    page_texts = get_pdf_page_texts("sample.pdf")
    print(f"Total pages loaded: {len(page_texts)}")

    # 2) Chunk them into DocumentChunk objects
    doc_id = "sample_doc"
    chunks = chunk_pdf_pages_to_document_chunks(doc_id, page_texts, max_words=200)

    print(f"Total chunks created: {len(chunks)}\n")

    # 3) Show a few example chunks
    for i, ch in enumerate(chunks[:5], start=1):
        print("=" * 60)
        print(f"Chunk {i}")
        print(f" - doc_id: {ch.doc_id}")
        print(f" - page: {ch.page}")
        print(f" - modality: {ch.modality}")
        print(f" - content (first 300 chars):\n{ch.content[:300]}")
        print()

    # 4) Show how many chunks came from the first 3 pages (just for feel)
    page_chunk_counts = {}
    for ch in chunks:
        page_chunk_counts[ch.page] = page_chunk_counts.get(ch.page, 0) + 1

    print("\nChunk counts per page:")
    for page, count in sorted(page_chunk_counts.items()):
        print(f" - Page {page}: {count} chunks")


if __name__ == "__main__":
    main()
