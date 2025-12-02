from typing import List, Dict, Tuple
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

try:
    from .chunker import load_and_chunk_directory
    from .embeddings import EmbeddingModel
    from .vector_store import FaissVectorStore
except ImportError:
    # Fallback for direct script execution 
    from chunker import load_and_chunk_directory
    from embeddings import EmbeddingModel
    from vector_store import FaissVectorStore

load_dotenv()


class RAGEngine:
    """
    Core RAG engine for the Machine Learning Basics domain.
    Handles ingestion, retrieval, and answer generation.
    """

    def __init__(
        self,
        data_dir: str = "data/ml_basics",
        index_path: str = "faiss_index.bin",
        metadata_path: str = "metadata.json",
        embedding_model_name: str = "text-embedding-3-small",
        llm_model_name: str = "gpt-4o",
        similarity_threshold: float = 0.75,
    ):
        self.data_dir = data_dir
        self.embedder = EmbeddingModel(model_name=embedding_model_name)
        self.vector_store = FaissVectorStore(
            index_path=index_path,
            metadata_path=metadata_path,
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for LLM. Please set it in a .env file or environment.")

        self.llm_client = OpenAI(api_key=api_key)
        self.llm_model_name = llm_model_name
        self.similarity_threshold = similarity_threshold

    # ---------- Ingestion ----------

    def ingest(self, chunk_size: int = 300, chunk_overlap: int = 60) -> None:
        """
        Ingest the ML basics data directory, chunk, embed, and build vector store.
        """
        print(f"[RAGEngine] Ingesting data from: {self.data_dir}")
        chunks = load_and_chunk_directory(
            self.data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks found. Check data directory and .txt files.")

        texts = [c["text"] for c in chunks]
        print(f"[RAGEngine] Creating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.build(embeddings, chunks)
        self.vector_store.save()
        print("[RAGEngine] Ingestion and index building completed.")

    # ---------- Retrieval ----------

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Embed query and search in vector store.
        Returns list of (metadata_with_text, similarity).
        """
        if not self.vector_store.is_built():
            # Try to load from disk
            self.vector_store.load()

        query_emb = self.embedder.embed_text(query)
        results = self.vector_store.search(query_emb, k=k)

        # Simple de-duplication for overlapping chunks
        seen = set()
        unique_results: List[Tuple[Dict, float]] = []
        for meta, score in results:
            key = (meta["source"], meta["chunk_id"])
            if key not in seen:
                seen.add(key)
                unique_results.append((meta, score))

        return unique_results

    # ---------- Generation ----------

    def _build_context_block(self, retrieved: List[Tuple[Dict, float]]) -> str:
        """
        Build a text block from retrieved chunks for the LLM prompt.
        """
        blocks: List[str] = []
        for meta, score in retrieved:
            block = (
                f"[source={meta['source']}, chunk_id={meta['chunk_id']}, "
                f"similarity={score:.3f}]\n{meta['text']}"
            )
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    def answer_question(self, query: str, top_k: int = 5) -> str:
        """
        High-level method: retrieve relevant chunks and query the LLM.
        """
        retrieved = self.retrieve(query, k=top_k)
        if not retrieved:
            return "I could not retrieve any relevant context from the documentation."

        best_score = max(score for _, score in retrieved)
        if best_score < self.similarity_threshold:
            return (
                "The documentation does not contain enough relevant information "
                "to answer this question confidently."
            )

        context_block = self._build_context_block(retrieved)

        system_prompt = (
            "You are an AI assistant specialized in Machine Learning Basics.\n"
            "You must answer only using the provided context chunks from the documentation.\n"
            "If the answer is not clearly contained in the context, say you do not know based on the documents.\n"
            "Do not invent facts or rely on outside knowledge.\n"
            "Always mention which source files you used in your answer."
        )

        user_message = f"""CONTEXT:
---------
{context_block}

---------
USER QUESTION:
{query}

Instructions:
- Use only the information from the CONTEXT.
- If the context is insufficient, say:
  "I do not know based on the provided documentation."
- If multiple sources conflict, say that there are conflicting descriptions and explain both.
"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()

        # Add simple source highlighting at the end:
        sources = {
            (meta["source"], meta["chunk_id"]) for meta, _ in retrieved
        }
        source_lines = [
            f"- {src} (chunk {cid})" for (src, cid) in sorted(sources)
        ]
        answer += "\n\nSources used:\n" + "\n".join(source_lines)
        return answer
