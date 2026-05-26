import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULTS = {
    "city": {
        "input": Path("data/raw/wikivoyage_pages.jsonl"),
        "output": Path("data/processed/travel_corpus.jsonl"),
    },
    "itineraries": {
        "input": Path("data/raw/wikivoyage_itineraries.jsonl"),
        "output": Path("data/processed/itinerary_corpus.jsonl"),
    },
    "pdf": {
        "input": Path("data/raw/pdf_guides.jsonl"),
        "output": Path("data/processed/pdf_corpus.jsonl"),
    },
}

def split_text_into_chunks(text: str, max_words: int = 180) -> List[str]:
    words = text.split()
    chunks = []

    for start in range(0, len(words), max_words):
        chunk_words = words[start : start + max_words]
        chunk = " ".join(chunk_words).strip()

        if len(chunk.split()) >= 40:
            chunks.append(chunk)

    return chunks

def make_training_prompt(record: Dict, chunk: str) -> str:
    return (
        f"Country: {record['country']}\n"
        f"City: {record['city']}\n"
        f"Category: {record['category']}\n"
        f"Section: {record['section']}\n"
        f"Travel information: {chunk}"
    )

def build_processed_record(
    mode: str, record: Dict, chunk_id: int, chunk: str
) -> Dict:
    if mode == "city":
        return {
            "source": record["source"],
            "url": record["url"],
            "country": record["country"],
            "city": record["city"],
            "section": record["section"],
            "category": record["category"],
            "chunk_id": chunk_id,
            "text": chunk,
            "training_text": make_training_prompt(record, chunk),
        }
    elif mode == "itineraries":
        return {
            "source": record["source"],
            "url": record["url"],
            "itinerary_slug": record["itinerary_slug"],
            "section": record["section"],
            "category": record["category"],
            "chunk_id": chunk_id,
            "text": chunk,
            "training_text": (
                f"Country: \n"
                f"City: \n"
                f"Category: {record['category']}\n"
                f"Section: {record['section']}\n"
                f"Travel information: {chunk}"
            ),
        }
    elif mode == "pdf":
        
        return {
            "source": record["source"],
            "source_name": record["source_name"],
            "country": record["country"],
            "city": record["city"],
            "section": record["section"],
            "category": record["category"],
            "page_number": record["page_number"],
            "chunk_id": record["chunk_id"],
            "text": chunk,
            "training_text": make_training_prompt(record, chunk),
        }

def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk Wikivoyage JSONL for training or downstream use.")
    parser.add_argument(
        "--mode",
        choices=("city", "itineraries", "pdf"),
        required=True,
        help="city: wikivoyage_pages with training_text; itineraries: wikivoyage_itineraries, text chunks only.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override input JSONL (default depends on --mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output JSONL (default depends on --mode).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=180,
        help="Max words per chunk (same splitter for both modes).",
    )
    args = parser.parse_args()

    raw_path: Path = args.input or DEFAULTS[args.mode]["input"]
    processed_path: Path = args.output or DEFAULTS[args.mode]["output"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    output_records: List[Dict] = []

    with raw_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            chunks = split_text_into_chunks(record["text"], max_words=args.max_words)

            for chunk_id, chunk in enumerate(chunks):
                output_records.append(
                    build_processed_record(args.mode, record, chunk_id, chunk)
                )

    with processed_path.open("w", encoding="utf-8") as output_file:
        for rec in output_records:
            output_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(output_records)} processed chunks to {processed_path}")

if __name__ == "__main__":
    main()