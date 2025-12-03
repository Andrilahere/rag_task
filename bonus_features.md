# Bonus Features 

This document explains the bonus features associated with the RAG system assignment.

# ✅ Implemented Bonus Feature

## 1. **Source Highlighting**

**Location:** `src/rag_engine.py` → inside `answer_question()` at the end.

###  What the Code Does

* Collects `(source, chunk_id)` pairs from retrieved chunks
* Deduplicates them using a `set()`
* Appends a formatted “Sources used” section at the end of the LLM-generated answer

### Code Snippet (Already Implemented)

```python
# Add simple source highlighting at the end:
sources = {
    (meta["source"], meta["chunk_id"]) for meta, _ in retrieved
}
source_lines = [
    f"- {src} (chunk {cid})" for (src, cid) in sorted(sources)
]
answer += "\n\nSources used:\n" + "\n".join(source_lines)
```

### Difference from a Plain RAG System

A standard RAG pipeline returns only the LLM answer.
This upgraded version **exposes the exact document chunks** used for grounding the answer.
This improves trust, transparency, and explainability — matching the *Source Highlighting* bonus requirement.

---

# Bonus Features That Can Be Implemented

# 2. **Metadata Filtering**

**Where it fits:** `src/rag_engine.py` → inside `retrieve()`

Each chunk already contains metadata:

* `source`
* `chunk_id`
* `start_word`
* `end_word`

### Pseudo-Code Implementation

```
def retrieve(self, query: str, k: int = 5, filter_sources: Optional[List[str]] = None):
    results = self.vector_store.search(query_emb, k=k)
    filtered = []
    for meta, score in results:
        if filter_sources and meta["source"] not in filter_sources:
            continue
        filtered.append((meta, score))
    return filtered
```

### How This Would Work

* Users select which files to include (e.g., via Streamlit UI)
* Retrieval only returns chunks whose metadata match the filters

---

# 3. **Multi-Query Retrieval**

* Each chunk already has metadata fields like: source (filename), chunk_id, start_word, end_word, etc.

**Where it fits:**

* RAGEngine.retrieve(...) in src/rag_engine.py.

### Pseudo-Code

#### Generate alternative queries

```python
def _generate_alternative_queries(self, query: str) -> List[str]:
    # using LLM to create 2-3 paraphrases
    ...

def retrieve(self, query: str, k: int = 5, use_multi_query: bool = False):
    queries = [query]
    if use_multi_query:
        queries.extend(self._generate_alternative_queries(query))

    embeddings = self.embedder.embed_texts(queries)

    aggregated = {}
    for emb in embeddings:
        results = self.vector_store.search(emb, k=k)
        for meta, score in results:
            key = (meta["source"], meta["chunk_id"])
            if key not in aggregated or score > aggregated[key][1]:
                aggregated[key] = (meta, score)

    return sorted(aggregated.values(), key=lambda x: x[1], reverse=True)[:k]
```

### How This Improves the System

* Multiple query embeddings increase the probability of retrieving relevant chunks
* Great for short or ambiguous user questions


# 4. **Context Compression**

**Where it fits:**

* New helper: `_compress_context()`
* Modify `answer_question()` to toggle compression

### Pseudo-Code for Compression

```python
def _compress_context(self, retrieved: List[Tuple[Dict, float]]) -> str:
    all_text = "\n\n".join(meta["text"] for meta, _ in retrieved)
    prompt = (
        "Summarize the key points from the following documentation chunks "
        "so that another model can answer a question based on the summary.\n\n"
        f"{all_text}"
    )
    response = self.llm_client.chat.completions.create(
        model=self.llm_model_name,
        messages=[
            {"role": "system", "content": "You compress context for downstream QA."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
```

###  Modified `answer_question()` usage

```python
if use_context_compression:
    context_for_prompt = self._compress_context(retrieved)
else:
    context_for_prompt = self._build_context_block(retrieved)
```



