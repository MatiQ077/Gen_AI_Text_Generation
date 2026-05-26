import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#from infer import generate  This line is used for tranformer but from scratch

import sys; sys.path.insert(0, "src_gpt")
from infer import generate

INDEX_PATH = Path("artifacts/rag_index.faiss")
META_PATH = Path("artifacts/rag_metadata.jsonl")

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_index = faiss.read_index(str(INDEX_PATH))
_metadata = [json.loads(l) for l in META_PATH.open(encoding="utf-8") if l.strip()]


def retrieve(query: str, k: int = 5) -> list:
    vec = _embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = _index.search(vec, k)
    return [_metadata[i] for i in indices[0] if i < len(_metadata)]


def _format_context(chunks: list, max_words_per_chunk: int = 60) -> str:
    excerpts = [" ".join(c["text"].split()[:max_words_per_chunk]) for c in chunks]
    return " ".join(excerpts)


def build_prompt(destination: str, trip_length: str, context: str = "") -> str:
    context_prefix = f"{context}\n" if context else ""
    return (
        f"Country: \nCity: {destination}\n"
        f"Category: itinerary\n"
        f"Section: {trip_length} trip\n"
        f"Travel information: {context_prefix}Day 1:\nMorning:"
    )


def generate_itinerary(destination, trip_length, budget, interests,
                       travel_style, pace, special_prefs) -> str:
    query = (
        f"{destination} {interests} {travel_style} {budget} {pace} "
        f"{special_prefs} travel itinerary"
    )
    chunks = retrieve(query, k=3)
    context = _format_context(chunks)
    prompt = build_prompt(destination, trip_length, context)
    print("=== PROMPT ===\n", prompt)
    return generate(prompt, max_new_tokens=250, temperature=0.8, top_p=0.9, repetition_penalty=1.5)