from app.ingestion import extract_image_ocr_chunks


def main():
    chunks = extract_image_ocr_chunks(pdf_name="sample.pdf", doc_id="sample_doc")

    print(f"Total image OCR chunks: {len(chunks)}\n")

    for i, ch in enumerate(chunks[:5], start=1):
        print("=" * 60)
        print(f"Chunk {i}")
        print(f" - page: {ch.page}")
        print(f" - modality: {ch.modality}")
        print(f" - content (first 300 chars):")
        print(ch.content[:300].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
