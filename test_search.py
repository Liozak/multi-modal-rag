from app.ingestion import get_pdf_page_texts
from app.chunking import chunk_pdf_pages_to_document_chunks
from app.index import build_text_index, search_text_index


def main():
    # 1) Load page texts from the PDF
    page_texts = get_pdf_page_texts("sample.pdf")
    print(f"Loaded {len(page_texts)} pages from PDF.\n")

    # 2) Chunk them into DocumentChunk objects
    doc_id = "sample_doc"
    chunks = chunk_pdf_pages_to_document_chunks(doc_id, page_texts, max_words=200)
    print(f"Created {len(chunks)} chunks.\n")

    # 3) Build the text index
    print("Building text index (this may download a model the first time)...\n")
    index = build_text_index(chunks)
    print("Index built successfully!\n")

    # 4) Try some example queries
    example_queries = [
        "What is this document about?",
        "Tell me something about inflation.",
        "What does it say about GDP or economic growth?",
    ]

    for q in example_queries:
        print("=" * 80)
        print(f"🔍 Query: {q}\n")

        results = search_text_index(index, q, top_k=3)

        if not results:
            print("No results found.\n")
            continue

        for rank, (score, chunk) in enumerate(results, start=1):
            print(f"Result {rank} | similarity = {score:.4f}")
            print(f" - doc_id: {chunk.doc_id}")
            print(f" - page: {chunk.page}")
            print(f" - modality: {chunk.modality}")
            print(" - content (first 300 chars):")
            print(chunk.content[:300].replace("\n", " "))
            print()

        print()  # extra spacing


if __name__ == "__main__":
    main()
