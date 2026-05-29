"""
scrape_reddit.py — Scrape structured travel itineraries from Reddit.

Searches r/travel, r/solotravel, and r/backpacking using Reddit's public JSON
API. Only self-posts that contain clear itinerary structure (a day reference + a time-of-day word) are kept.

The destination and trip length are parsed from the post title using a regex
so the training_text can be formatted consistently with the other corpora.

Output: data/raw/reddit_itineraries.jsonl
    Consumed by src_gpt/train.py as REDDIT_CORPUS.

Usage:
    python src/scrape_reddit.py
    python src/scrape_reddit.py --max-pages 5
    python src/scrape_reddit.py --out data/raw/reddit_itineraries.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

OUTPUT_PATH = Path("data/raw/reddit_itineraries.jsonl")
USER_AGENT  = "TravelItineraryStudentProject/1.0"

SUBREDDITS = ["travel", "solotravel", "backpacking"]
SEARCH_QUERIES = ["days in", "day itinerary", "trip report", "week in", "day by day"]

_DAY_RE  = re.compile(r'\bday\s*[1-9]\d?\b',  re.IGNORECASE)
_TIME_RE = re.compile(r'\b(morning|afternoon|evening|night|lunch|dinner|breakfast)\b', re.IGNORECASE)
_URL_RE  = re.compile(r'https?://\S+')
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')  # [text](url) → text
_BOLD    = re.compile(r'\*{1,3}')
_HEADER  = re.compile(r'(?m)^#{1,6}\s*')

def has_itinerary_structure(text: str) -> bool:
    return bool(_DAY_RE.search(text) and _TIME_RE.search(text))

def clean_text(text: str) -> str:    
    text = _MD_LINK.sub(r'\1', text)
    text = _URL_RE.sub('', text)
    text = _BOLD.sub('', text)
    text = _HEADER.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_destination(title: str) -> tuple[str, str]:
    
    m = re.search(
        r'(\d+)\s*(day|days|week|weeks?)\s*(?:in|to|for|across|through|around)\s+'
        r'([A-Za-z][A-Za-z\s,]{2,35})',
        title, re.IGNORECASE,
    )
    if m:
        count = int(m.group(1))
        unit  = m.group(2).lower()
        dest  = m.group(3).strip().rstrip('.,!?-(')
        if 'week' in unit:
            count *= 7
        return dest, f"{count} day trip"
    return "", "itinerary"

def fetch_page(subreddit: str, query: str, after: str = "") -> Optional[dict]:
 
    params: dict = {
        "q":           query,
        "sort":        "top",
        "t":           "all",
        "limit":       100,
        "restrict_sr": "true",
    }
    if after:
        params["after"] = after
    try:
        resp = requests.get(
            f"https://www.reddit.com/r/{subreddit}/search.json",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [ERROR] r/{subreddit} '{query}': {exc}")
        return None

# Look for subreddit, query pair
def scrape_pair(subreddit: str, query: str, max_pages: int) -> list[dict]:
    
    records: list[dict] = []
    after = ""

    for _ in range(max_pages):
        data = fetch_page(subreddit, query, after)
        if data is None:
            break

        posts = data.get("data", {}).get("children", [])
        if not posts:
            break

        for post in posts:
            p = post["data"]

            if not p.get("is_self"):
                continue

            body = p.get("selftext", "").strip()
            if body in ("[deleted]", "[removed]", ""):
                continue

            body = clean_text(body)

            if not has_itinerary_structure(body):
                continue
            if len(body.split()) < 80:
                continue

            title         = p.get("title", "")
            dest, section = extract_destination(title)
            training_text = (
                f"Country: \nCity: {dest}\n"
                f"Category: itinerary\n"
                f"Section: {section}\n"
                f"Travel information: {body}"
            )

            records.append({
                "source":        "reddit",
                "subreddit":     subreddit,
                "url":           f"https://www.reddit.com{p.get('permalink', '')}",
                "title":         title,
                "city":          dest,
                "country":       "",
                "section":       section,
                "category":      "itinerary",
                "text":          body,
                "training_text": training_text,
            })

        after = data.get("data", {}).get("after", "")
        if not after:
            break
        time.sleep(1.5)

    return records

def main() -> None:    
    parser = argparse.ArgumentParser(description="Scrape travel itineraries from Reddit.")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max-pages", type=int, default=3, help="Pages to fetch per (subreddit, query) pair — 100 posts per page.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen_urls: set[str]  = set()
    all_records: list[dict] = []

    pairs = [(sr, q) for sr in SUBREDDITS for q in SEARCH_QUERIES]
    for subreddit, query in tqdm(pairs, desc="Scraping Reddit"):
        for rec in scrape_pair(subreddit, query, args.max_pages):
            if rec["url"] not in seen_urls:
                seen_urls.add(rec["url"])
                all_records.append(rec)
        time.sleep(2.0)

    with args.out.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved {len(all_records)} itinerary posts → {args.out}")

if __name__ == "__main__":
    main()