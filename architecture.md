# RAG System Architecture Document

## 1. System Overview
A Retrieval-Augmented Generation system that answers queries about Machine Learning Basics using document retrieval and LLM generation.

It follows the classic RAG pattern:

1. **Ingestion**: Load domain documents → clean text → chunk with overlap → embed → store in a vector index.
2. **Retrieval**: Embed query → similarity search in vector store → get top-k chunks.
3. **Generation**: Pass retrieved chunks + user query to an LLM with a constrained prompt to generate an answer grounded in the retrieved context.


RAG Architecture Diagram:

- Attached in the Word File

## 2. Text Extraction & Chunking Plan

Document Format
The dataset consists of multiple .txt files containing machine learning basics (supervised learning, unsupervised learning, overfitting, regularization, etc.).

Text Cleaning Applied

•	Normalize newline format (\r\n → \n)
•	Remove extra blank lines
•	Collapse multiple spaces/tabs
•	Strip leading/trailing whitespace

Chunking Strategy

Type: Word-level chunking with overlap
•	Chunk size: 300 words
•	Overlap: 60 words

Why This Works

•	Avoids losing context across chunk boundaries
•	Works well with small datasets
•	Reduces LLM hallucination risk because related concepts remain grouped
•	Improves retrieval recall

## 3. Embedding Model Choice & Justification

Model Used:
OpenAI text-embedding-3-small

Justification:

•	High quality embeddings suitable for semantic search
•	Low cost compared to larger models
•	768-dimension vectors give good performance for RAG tasks
•	Works seamlessly with FAISS
•	Ideal for small-to-medium RAG projects

The chosen model strikes the best balance of:

•	Cost
•	Speed
•	Accuracy

## 4. Vector Database Choice

Vector Store Used:
FAISS
Reasons for Choosing FAISS

•	Very fast local similarity search
•	Lightweight and easy to integrate
•	Works offline
•	Ideal for small datasets (no need for cloud vector DB)
•	Supports multiple index types (L2, cosine, IVF, HNSW)

Structure

•	faiss_index.bin - stores the vector index
•	metadata.json - stores metadata for each chunk: source, chunk_id, start_word, end_word, text

## 5. Retrieval Strategy

Steps:

1.	Embedding the user query using the same embedding model.
2.	Using FAISS to obtain the top-k most similar chunks (default k=5).
3.	De-duplicate overlapping chunks by (source, chunk_id).
4.	Appling a similarity threshold check: if best_score < 0.75 → return fallback message
5.	Building a context block including:
•	chunk text
•	source file
•	chunk ID
•	similarity score

Why This Strategy Works

•	Ensures only highly relevant chunks are passed to the LLM
•	Prevents hallucination from weak matches
•	Keeps prompt size manageable

## 6. Prompt Structure

System Prompt:

Guides the LLM to answer only from the documentation:
You are an AI assistant specialized in Machine Learning Basics. You must answer only using the provided context chunks from the documentation.
If the answer is not clearly contained in the context, say:
"I do not know based on the provided documentation."
Do not invent facts or rely on outside knowledge.
Always mention which source files used in the answer.

User Prompt Template:
CONTEXT:
---------
{context_block}

---------
USER QUESTION:
{query}

Instructions:

•	Use only the information from the CONTEXT.
•	If the context is insufficient, say: "I do not know based on the provided documentation."
•	If multiple sources conflict, say that there are conflicting descriptions.

Why This Prompting Works

•	Strong grounding instructions
•	Forces fallback instead of hallucinating
•	Encourages citing specific source chunks

## 7. Failure Modes & Hallucination Prevention

Failure Modes

1.	Low-similarity retrieval - irrelevant results returned
2.	Contradictory content - LLM confusion
3.	Very broad queries - too many chunks retrieved
4.	Chunk fragmentation - missing relationships
   
Mitigations Implemented

•	Similarity threshold (0.75)
•	Overlapping chunks
•	Strict system prompt
•	Explicit fallback responses when context is insufficient
•	Source highlighting at the end of the answer

Mitigations Planned (Bonus Section)

•	Metadata filtering
•	Multi-query retrieval
•	Context compression
•	LLM-based re-ranking

## 8. Evaluation Plan

Qualitative Evaluation
Run a set of representative queries covering:

•	Definitions
•	Differences between ML concepts
•	Advantages/disadvantages
•	Examples

Measure:

•	Correctness
•	Grounding
•	Consistency across queries

Stress Testing

•	Ambiguous queries
•	Multi-hop reasoning
•	Out-of-domain questions
