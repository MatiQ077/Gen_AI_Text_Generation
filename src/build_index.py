import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CORPORA = [
    Path("data/processed/travel_corpus.jsonl"),
    Path("data/processed/itinerary_corpus.jsonl"),
    Path("data/processed/pdf_corpus.jsonl"),
]
INDEX_PATH = Path("artifacts/rag_index.faiss")
META_PATH = Path("artifacts/rag_metadata.jsonl")


def load_chunks(paths):
    chunks = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def main():
    print("Loading chunks...")
    chunks = load_chunks(CORPORA)
    print(f"Total chunks: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]

    print("Embedding chunks (this takes a few minutes)...")
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = embeddings.astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved index ({index.ntotal} vectors) → {INDEX_PATH}")

    with META_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            meta = {
                "text": chunk["text"],
                "country": chunk.get("country", ""),
                "city": chunk.get("city", ""),
                "category": chunk.get("category", ""),
                "section": chunk.get("section", ""),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"Saved metadata → {META_PATH}")


if __name__ == "__main__":
    main()
