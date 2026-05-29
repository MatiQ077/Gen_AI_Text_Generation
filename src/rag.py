"""
rag.py retrieve relevant chunks, build a prompt, generate.

This module is a connection between user input (destination + preferences), the
FAISS retrieval index, and the text generator.  It is imported by app.py and
exposes a single public function: generate_itinerary().

Pipeline for one request:
    1. Build a free-text query from all user preferences.
    2. Embed the query with the same sentence-transformer used to build the index.
    3. k-NN search in FAISS → top-k metadata chunks.
    4. Truncate and concatenate chunks into a context string.
    5. Inject context into the structured training-format prompt.
    6. Call generate() from the active inference backend.

The FAISS index, metadata, and embedder are loaded once at import time so
repeated calls from the Gradio UI do not reload them.

Active inference backend: src_gpt/infer.py (GPT-based model).
Alternate backend:        src/infer.py    (from-scratch transformer — see comment below).
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


#from infer import generate    # Two inference backends exist.  Uncomment the first line to switch to the
sys.path.insert(0, "src_gpt")
from infer import generate

INDEX_PATH = Path("artifacts/rag_index.faiss")
META_PATH  = Path("artifacts/rag_metadata.jsonl")

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_index    = faiss.read_index(str(INDEX_PATH))
_metadata = [json.loads(l) for l in META_PATH.open(encoding="utf-8") if l.strip()]

#Embed a query and return the top-k nearest chunks from the FAISS index
def retrieve(query: str, k: int = 5) -> list:
    
    vec = _embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = _index.search(vec, k)
    return [_metadata[i] for i in indices[0] if i < len(_metadata)]


def format_context(chunks: list, max_words_per_chunk: int = 60) -> str:
   
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


#Generate itinerary using RAG
def generate_itinerary(
    destination: str,
    trip_length: str,
    budget: str,
    interests: str,
    travel_style: str,
    pace: str,
    special_prefs: str,
) -> str:
    
    query = (
        f"{destination} {interests} {travel_style} {budget} {pace} "
        f"{special_prefs} travel itinerary"
    )
    chunks  = retrieve(query, k=3)
    context = format_context(chunks)
    prompt  = build_prompt(destination, trip_length, context)

    print("=== PROMPT ===\n", prompt)

    return generate(prompt, max_new_tokens=250, temperature=0.8, top_p=0.9, repetition_penalty=1.5)