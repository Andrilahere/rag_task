## 1. What worked well?

# a. Clean and modular architecture
The pipeline separated concerns clearly:
•	chunker.py - text cleaning + chunking
•	embeddings.py - embedding model wrapper
•	vector_store.py - FAISS-based vector storage
•	rag_engine.py - retrieval + reasoning + LLM orchestration
•	app.py - Streamlit UI

# b. RAG grounding worked reliably
For most questions related to supervised learning, unsupervised learning, overfitting, etc., the system:
•	retrieved the correct chunks
•	provided grounded answers
•	linked the chunks through source highlighting

# c. Overlapping chunk strategy performed well
The overlap of 60 tokens minimized context fragmentation and ensured definitions and examples stayed connected.

# d. FAISS performed efficiently
Even with a small dataset, FAISS’s vector search behaved consistently and gives fast nearest neighbors locally.

## 2. What didn’t work?

a. LLM sometimes ignores instructions and hallucinates; enforcement isn’t perfect.
b. Contradictory content across documents can confuse the LLM if multiple chunks contain differing claims.
c. For multi-hop queries requiring reasoning across many chunks, naive concatenation causes token bloat; need a compression/summarization step.
d. Small demo uses FAISS and local metadata store - lacks advanced filtering and vector versioning found in production DBs.

## 3. How to scale to 100,000 documents?

Scaling requires addressing storage, retrieval latency, ingestion speed, and memory management.

# a. Efficient Vector Indexing
•	Replace FAISS flat index (IndexFlatL2) with:FAISS HNSW or FAISS IVF (Inverted File Index)
These reduce search time from O(n) to approx. O(log n).

# b. Incremental ingestion pipeline
For large corpora:
•	Stream documents
•	Embed in batches
•	Insert into FAISS incrementally
•	Avoid re-building the entire index each time

# c. Store vectors + metadata in a scalable vector DB
Moving to Pinecone, Weaviate, Qdrant, Milvus which offers horizontal scaling, distributed search, pagination, filtering
# d. Caching
Adding caching for popular queries and asynchronous ingestion pipelines. This reduces load on both FAISS and the LLM.

## 4. How to reduce hallucinations?

# a. Strong system prompts or better prompt engineering
•	“Using ONLY the provided context.”
•	“If unsure, respond: I do not know based on the provided documentation.”
This helped but not fully eliminated hallucination.

# b. Reduce chunk size for precision
Smaller chunks = more accurate retrieval.
Overlapped chunks preserve context while improving specificity.

# c. Re-rank with LLM
Using an LLM to score relevance and factual grounding before answering.

# d. Answer verification
Implementing multi-hop verification across chunks.
Also, running the final answer through another LLM with the instruction:
“Verify that the answer is strictly grounded in the provided context.”

## 5. What to improve with more time?

# a. Implementation of advanced retrieval enhancements
•	Multi-query retrieval
•	Re-ranking
•	Query expansion
•	Context compression
These would significantly improve answer accuracy and reduce prompt size.

# b. Structured evaluation
•	Implement automated testing with diverse queries
•	Add quantitative metrics (BLEU, ROUGE, METEOR)
•	Create benchmark dataset

# c. Better UI with chunk-level visualization
Adding:
•	color-coded similarity heatmaps
•	chunk preview panel
•	document explorer

# d. Domain routing
If more domains are added (Finance, HR, Legal), a router model needs to direct queries to the correct vector store.

# e. Deployment
Building:
•	Dockerfile
•	FastAPI wrapper
•	CI/CD pipeline
•	Cloud deployment (Render / AWS / GCP)

# f. Self-improving System:
•	Active learning from user interactions
•	Automatic chunk optimization
•	Dynamic threshold adjustment




