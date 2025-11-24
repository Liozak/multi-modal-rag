import sys
from pathlib import Path

# ✅ Add project root to Python path so "app" can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from app.qa_pipeline import build_qa_index_from_pdf, answer_question

st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")

st.title("📄 Multi-Modal RAG QA System")
st.write("Ask questions about your uploaded PDF document.")

# Build index only once
@st.cache_resource
def load_index():
    with st.spinner("Building index from Qatar IMF PDF... (first time only)"):
        index = build_qa_index_from_pdf(pdf_name="qatar_test_doc.pdf", doc_id="qatar_report")
    return index


# Initialize index
index = load_index()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_query = st.text_input("Ask a question about the document:")

if st.button("Ask") and user_query.strip():

    with st.spinner("Thinking..."):
        answer, chunks = answer_question(index, user_query, top_k=5)

    # Save chat
    st.session_state.chat_history.append((user_query, answer, chunks))

# Display chat history
for i, (q, a, cks) in enumerate(reversed(st.session_state.chat_history), 1):
    st.markdown(f"### 🧑‍💻 You:\n{q}")
    st.markdown(f"### 🤖 Answer:\n{a}")

    with st.expander("📍 Source chunks used"):
        for ch in cks:
            st.markdown(
                f"""
**Page {ch.page} | {ch.modality}**  
{ch.content[:500]}...
""",
                unsafe_allow_html=True
            )

    st.markdown("---")
