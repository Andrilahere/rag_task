from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class EmbeddingModel:
    """
    Wrapper around OpenAI embedding model.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing in .env or environment.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
