"""
scrape_stackexchange.py — Scrape Travel Stack Exchange Q&A via the public API.

Fetches top-voted questions and their highest-scored answers for a fixed set
of travel-related tags.  Each (question, answer) pair is saved as a single
record with a training_text field formatted to match the rest of the corpus.

Output: data/raw/stackexchange_itineraries.jsonl

Usage:
    python src/scrape_stackexchange.py
    python src/scrape_stackexchange.py --key YOUR_API_KEY --max-pages 50

"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

API_BASE    = "https://api.stackexchange.com/2.3"
SITE        = "travel"
OUTPUT_PATH = Path("data/raw/stackexchange_itineraries.jsonl")

TAGS = [
    "trip-planning",
    "day-trips",
    "sightseeing",
    "recommendations",
    "planning",
    "budget",
    "food-and-drink",
    "accommodation",
    "tips-and-tricks",
]

def strip_html(html: str) -> str:
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split()).strip()

def api_get(path: str, params: dict, api_key: Optional[str]) -> dict:
    params = {"site": SITE, **params}
    if api_key:
        params["key"] = api_key
    resp = requests.get(f"{API_BASE}/{path}", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

# Fetch top voted question from a page
def fetch_questions(tag: str, page: int, pagesize: int, api_key: Optional[str]) -> dict:
    
    return api_get("questions", {
        "tagged":   tag,
        "sort":     "votes",
        "order":    "desc",
        "pagesize": pagesize,
        "page":     page,
        "filter":   "withbody",
    }, api_key)

def fetch_top_answers(question_ids: List[int], api_key: Optional[str]) -> Dict[int, dict]:
    ids_str = ";".join(str(i) for i in question_ids)
    data    = api_get(f"questions/{ids_str}/answers", {
        "sort":     "votes",
        "order":    "desc",
        "filter":   "withbody",
        "pagesize": 100,
    }, api_key)
    best: Dict[int, dict] = {}
    for answer in data.get("items", []):
        qid = answer["question_id"]
        if qid not in best or answer["score"] > best[qid]["score"]:
            best[qid] = answer
    return best

def make_training_text(q_text: str, a_text: str) -> str:
    return (
        f"Country: \nCity: \nCategory: itinerary\n"
        f"Section: trip planning\n"
        f"Travel information: {q_text} {a_text}"
    )

def scrape_tag(
    tag: str,
    max_pages: int,
    pagesize: int,
    sleep: float,
    api_key: Optional[str],
) -> List[Dict]:
    
    records = []
    for page in range(1, max_pages + 1):
        data      = fetch_questions(tag, page, pagesize, api_key)
        questions = [q for q in data.get("items", []) if q.get("answer_count", 0) > 0]

        if questions:
            question_ids = [q["question_id"] for q in questions]
            top_answers  = fetch_top_answers(question_ids, api_key)
            time.sleep(sleep)

            for q in questions:
                qid = q["question_id"]
                if qid not in top_answers:
                    continue
                q_text = strip_html(q.get("body", ""))
                a_text = strip_html(top_answers[qid].get("body", ""))
                if len(q_text) < 50 or len(a_text) < 100:
                    continue
                records.append({
                    "source":        "travel_stackexchange",
                    "url":           q.get("link", ""),
                    "question_id":   qid,
                    "tags":          q.get("tags", []),
                    "text":          f"{q_text} {a_text}",
                    "training_text": make_training_text(q_text, a_text),
                })

        if not data.get("has_more"):
            break
        time.sleep(sleep)

    return records

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Travel Stack Exchange itinerary Q&A via the public API."
    )
    parser.add_argument(
        "--max-pages", type=int, default=5,
        help="Pages per tag (100 questions/page). Default 5 = up to 500 per tag.",
    )
    parser.add_argument("--pagesize", type=int, default=100)
    parser.add_argument(
        "--sleep", type=float, default=1.5,
        help="Seconds between requests.",
    )
    parser.add_argument(
        "--key", type=str, default=None,
        help="Stack Exchange API key (free at stackapps.com). Raises daily limit 300 → 10 000.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    seen_ids: set      = set()
    all_records: List[Dict] = []

    for tag in tqdm(TAGS, desc="Tags"):
        records = scrape_tag(tag, args.max_pages, args.pagesize, args.sleep, args.key)
        new = 0
        for rec in records:
            if rec["question_id"] not in seen_ids:
                seen_ids.add(rec["question_id"])
                all_records.append(rec)
                new += 1
        print(f"  {tag}: {new} new records (total unique: {len(all_records)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved {len(all_records)} unique Q&A records to {args.output}")

if __name__ == "__main__":
    main()