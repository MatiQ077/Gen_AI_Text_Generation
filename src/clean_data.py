import json
from pathlib import Path
from typing import Dict, List

RAW_PATH = Path("data/raw/wikivoyage_pages.jsonl")
PROCESSED_PATH = Path("data/processed/travel_corpus.jsonl")

def split_text_into_chunks(text: str, max_words: int = 180) -> List[str]:
   
    words = text.split()
    chunks = []

    for start in range(0, len(words), max_words):
        chunk_words = words[start:start + max_words]
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

def main() -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_records = []

    with RAW_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)

            chunks = split_text_into_chunks(record["text"])

            for chunk_id, chunk in enumerate(chunks):
                processed_record = {
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
                output_records.append(processed_record)

    with PROCESSED_PATH.open("w", encoding="utf-8") as output_file:
        for record in output_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(output_records)} processed chunks to {PROCESSED_PATH}")

if __name__ == "__main__":
    main()