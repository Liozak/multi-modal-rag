from typing import List, Dict

from app.qa_pipeline import build_qa_index_from_pdf, answer_question


# 🔎 Qatar IMF report – evaluation questions
# You can tweak keywords after you read the report more carefully.
EVAL_QUESTIONS: List[Dict] = [
    {
        "id": "q1_overview",
        "question": "What is the overall macroeconomic outlook for Qatar according to this report?",
        "expected_keywords": ["growth", "gdp", "outlook"],
    },
    {
        "id": "q2_growth_drivers",
        "question": "What are the main drivers of Qatar's recent and projected economic growth?",
        "expected_keywords": ["gas", "lng", "hydrocarbon", "non-hydrocarbon"],
    },
    {
        "id": "q3_inflation",
        "question": "What does the report say about inflation dynamics and price pressures in Qatar?",
        "expected_keywords": ["inflation", "prices", "inflationary"],
    },
    {
        "id": "q4_fiscal_policy",
        "question": "How does the report describe Qatar's fiscal position and fiscal policy stance?",
        "expected_keywords": ["fiscal", "surplus", "deficit", "consolidation"],
    },
    {
        "id": "q5_banking_sector",
        "question": "What assessment does the report give of Qatar's banking and financial sector?",
        "expected_keywords": ["banking", "capital", "liquidity", "NPL"],
    },
    {
        "id": "q6_external_position",
        "question": "How does the report describe Qatar's external position, current account, and reserves?",
        "expected_keywords": ["current account", "reserves", "external", "exports"],
    },
    {
        "id": "q7_risks",
        "question": "What are the key downside risks or uncertainties identified for Qatar's economy?",
        "expected_keywords": ["risk", "uncertainty", "volatility"],
    },
    {
        "id": "q8_diversification_reforms",
        "question": "What structural reforms and diversification efforts are highlighted in the report?",
        "expected_keywords": ["diversification", "reforms", "vision", "private sector"],
    },
    {
        "id": "q9_labor_market",
        "question": "What does the report mention about Qatar's labor market and employment conditions?",
        "expected_keywords": ["labor", "employment", "jobs", "participation"],
    },
    {
        "id": "q10_climate_transition",
        "question": "Does the report discuss climate-related policies or the energy transition for Qatar?",
        "expected_keywords": ["climate", "emissions", "transition", "decarbonization"],
    },
]


def main():
    print("📚 Building QA index from qatar_test_doc.pdf ...\n")
    index = build_qa_index_from_pdf(pdf_name="qatar_test_doc.pdf", doc_id="qatar_report")
    print("\n✅ Index ready. Starting evaluation...\n")

    total = len(EVAL_QUESTIONS)
    passed = 0

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        qid = item["id"]
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        print("=" * 80)
        print(f"🧪 Test {i}/{total} | ID: {qid}")
        print(f"❓ Question: {question}\n")

        # Run RAG QA
        answer, chunks = answer_question(index, question, top_k=5)

        print("🧠 Answer:")
        print(answer)
        print()

        # Modalities + pages used
        modalities = sorted({ch.modality for ch in chunks})
        pages = sorted({ch.page for ch in chunks})

        print(f"📄 Pages used: {pages}")
        print(f"🧬 Modalities used: {modalities}")

        # Simple keyword-based evaluation
        if expected_keywords:
            ans_lower = answer.lower()
            hits = [kw for kw in expected_keywords if kw.lower() in ans_lower]
            passed_test = len(hits) > 0
        else:
            passed_test = True
            hits = []

        if passed_test:
            print(f"✅ RESULT: PASS (keywords found: {hits})")
            passed += 1
        else:
            print(f"❌ RESULT: FAIL (none of {expected_keywords} found in answer)")

        print()

    print("=" * 80)
    print("📊 EVALUATION SUMMARY")
    print(f"Total questions: {total}")
    print(f"Passed (keyword-based): {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass rate: {passed}/{total} = {passed / total * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
