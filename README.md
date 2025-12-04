# **Mini RAG System – Machine Learning Basics**

This repository implements a **Retrieval-Augmented Generation (RAG)** system for the GFGC AI Intern Hiring Task.
The system answers questions strictly based on provided ML Basics documents using chunking, embeddings, vector similarity search, and an LLM.

---

#  **Project Structure**

```
rag_task/
├── README.md
├── architecture.md
├── bonus_features.md
├── evaluation.md
├── requirements.txt
├── data/ml_basics/
├── src/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_engine.py
│   └── app.py
```

Generated after ingestion:

```
faiss_index.bin
metadata.json
```

---

#  **Overview**

This RAG pipeline includes:

* **Text cleaning + chunking** (300-word chunks, 60 overlap)
* **OpenAI embeddings** (`text-embedding-3-small`)
* **FAISS vector store** for similarity search
* **RAG Engine** (retrieval + thresholding + prompting)
* **LLM answering** using GPT-4o
* **Streamlit UI** for interactive querying
* **Source highlighting** (bonus feature implemented)

---

#  **Setup**

## 1. Clone & Create Virtual Environment

```bash
git clone <repo-url>
cd rag_task
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Add API Key

Create `.env`:

```
OPENAI_API_KEY=your_key_here
```

---

#  **Build Index**

Automatically runs on first UI load.
To run manually:

```bash
python -m src.rag_engine ingest
```

---

#  **Run the Streamlit App**

```bash
streamlit run src/app.py
```

Features:

* Query input
* Retrieval settings
* Grounded answer
* Sources used
* Optional chunk preview

---

#  **Deliverables Included**

* `architecture.md` – full architecture (2 pages)
* `evaluation.md` – required evaluation note
* `bonus_features.md` – implemented & planned bonuses
* Complete source code
* ML basics dataset

---

#  **Bonus Features**

### ✔ Implemented

* **Source Highlighting** (shows filenames + chunk IDs)

###  Designed (Not implemented)

* Metadata filtering
* Multi-query retrieval
* Context compression
* LLM-based re-ranking

(Details in `bonus_features.md`)

---

#  **Example Questions**

* “What is supervised learning?”
* “Explain overfitting vs underfitting.”
* “Give examples of unsupervised learning.”

---

#  **Summary**

A complete, modular RAG system demonstrating:

* Clean architecture
* Reliable retrieval
* Hallucination reduction techniques
* Professional documentation
* Ready-to-extend bonus features

---

