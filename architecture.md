# RAG System Architecture Document

## 1. System Overview
A Retrieval-Augmented Generation system that answers queries about Machine Learning Basics using document retrieval and LLM generation.

It follows the classic RAG pattern:

1. **Ingestion**: Load domain documents → clean text → chunk with overlap → embed → store in a vector index.
2. **Retrieval**: Embed query → similarity search in vector store → get top-k chunks.
3. **Generation**: Pass retrieved chunks + user query to an LLM with a constrained prompt to generate an answer grounded in the retrieved context.
