# src/app.py

import streamlit as st
from pathlib import Path

# Support both package-style and direct imports depending on how Streamlit runs the app
try:
    from .rag_engine import RAGEngine
except ImportError:
    from rag_engine import RAGEngine


st.set_page_config(
    page_title="Machine Learning Basics Assistant",
    layout="wide",
)


def ensure_index(engine: RAGEngine) -> None:
    """
    Ensure that the FAISS index and metadata exist.
    If not, trigger ingestion to build them.
    """
    index_file = Path("faiss_index.bin")
    metadata_file = Path("metadata.json")

    if not index_file.exists() or not metadata_file.exists():
        with st.spinner("Building index from Machine Learning documents..."):
            engine.ingest()


def main() -> None:
    st.title("🧠 Mini RAG – Machine Learning Basics Assistant")
    st.write(
        """
        This app answers questions based on a small **Machine Learning Basics** corpus
        using a **Retrieval-Augmented Generation (RAG)** pipeline.
        """
    )

    # Optional small styling tweak
    st.markdown(
        """
        <style>
        .stTextInput>div>div>input {
            font-size: 16px;
        }
        .stButton>button {
            border-radius: 6px;
            padding: 0.4rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize backend engine
    engine = RAGEngine()
    ensure_index(engine)

    # Query input
    question = st.text_input(
        "🔍 Ask a question about Machine Learning basics:",
        placeholder="e.g., What is supervised learning?",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=10, value=5)
    with col2:
        ask = st.button("Get Answer")

    if ask and question.strip():
        with st.spinner("Retrieving relevant context and generating answer..."):
            try:
                answer = engine.answer_question(question, top_k=top_k)
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
                return

        st.markdown("### 📘 Answer")
        st.write(answer)

        # Try to parse and display the "Sources used" section nicely
        st.markdown("---")
        st.markdown("### 📂 Sources Used")

        # The RAGEngine appends:
        # "Sources used:\n- file (chunk id)\n- ..."
        lower_answer = answer.lower()
        src_pos = lower_answer.find("sources used")
        if src_pos != -1:
            sources_text = answer[src_pos:].split("\n", 1)[-1]
            for line in sources_text.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    st.markdown(line)
        else:
            st.info("No explicit source section found in the answer.")

        # Optional: show retrieved chunks for debugging / transparency
        st.markdown("---")
        show_chunks = st.checkbox("Show retrieved chunks (debug view)")
        if show_chunks:
            retrieved = engine.retrieve(question, k=top_k)
            if not retrieved:
                st.info("No chunks retrieved.")
            else:
                for meta, score in retrieved:
                    with st.expander(
                        f"{meta['source']} (chunk {meta['chunk_id']}) — similarity {score:.3f}"
                    ):
                        st.write(meta["text"])


if __name__ == "__main__":
    main()
