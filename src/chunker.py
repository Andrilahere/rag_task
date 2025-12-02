from typing import List, Dict
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Normalize newlines
    - Remove extra spaces
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text) 
    text = re.sub(r"[ \t]+", " ", text)    
    return text.strip()


def word_chunk(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> List[Dict]:
    """
    Split text into overlapping word chunks.

    Returns a list of dicts:
    {
        "chunk_id": int,
        "text": str,
        "start_word": int,
        "end_word": int
    }
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    chunks: List[Dict] = []

    if not words:
        return chunks

    step = chunk_size - chunk_overlap
    chunk_id = 0

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_word": start,
                "end_word": min(end, len(words)),
            }
        )
        chunk_id += 1
        if end >= len(words):
            break

    return chunks


def load_and_chunk_directory(
    data_dir: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> List[Dict]:
    """
    Load all .txt files from data_dir and return a list of chunk dicts
    with metadata: source filename + chunk info.
    """
    data_path = Path(data_dir)
    all_chunks: List[Dict] = []

    for file_path in data_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        file_chunks = word_chunk(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for ch in file_chunks:
            ch_with_meta = {
                "source": file_path.name,
                **ch,
            }
            all_chunks.append(ch_with_meta)

    return all_chunks

