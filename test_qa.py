from app.qa_pipeline import build_qa_index_from_pdf, answer_question


def main():
    print("Building QA index from qatar_test_doc.pdf ...\n")
    index = build_qa_index_from_pdf(pdf_name="qatar_test_doc.pdf", doc_id="qatar_report")

    print("Index built! You can now ask questions.\n")

    print("Type your question and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not query:
            print("Please type something.\n")
            continue

        print("\nThinking...\n")
        answer, chunks = answer_question(index, query, top_k=5)

        print("🧠 Answer:\n")
        print(answer)
        print("\n--- Context used (first 2 chunks) ---\n")

        for i, ch in enumerate(chunks[:2], start=1):
            print(f"[Chunk {i} | page={ch.page}]")
            print(ch.content[:400].replace("\n", " "))
            print()

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
