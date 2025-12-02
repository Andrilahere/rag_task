from typing import List, Dict, Tuple
import faiss
import numpy as np
import json
from pathlib import Path


class FaissVectorStore:
    """
    Simple FAISS-based vector store with metadata stored in JSON.
    """

    def __init__(
        self,
        index_path: str = "faiss_index.bin",
        metadata_path: str = "metadata.json",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None  # type: ignore
        self.metadata: List[Dict] = []

    def build(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        Build a new FAISS index from embeddings, and store metadata.
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided to build index.")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))
        self.metadata = metadata

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Index not built; cannot save.")
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Index or metadata file not found.")
        self.index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def is_built(self) -> bool:
        return self.index is not None and len(self.metadata) > 0

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Search top-k nearest vectors. Returns list of (metadata, similarity).
        Similarity is computed as 1 / (1 + L2_distance).
        """
        if self.index is None:
            raise ValueError("Index not loaded or built.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        distances = distances[0]
        indices = indices[0]

        results: List[Tuple[Dict, float]] = []
        for idx, dist in zip(indices, distances):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            similarity = float(1.0 / (1.0 + float(dist)))
            results.append((meta, similarity))
        return results
