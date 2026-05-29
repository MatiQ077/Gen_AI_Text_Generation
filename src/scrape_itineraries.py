"""
scrape_itineraries.py — Collect Wikivoyage itinerary articles and save as JSONL.

Two-phase scraper:
    Phase 1 — Discovery: fetch seed hub pages ("Itineraries", "Itineraries_index")
              and extract all outbound Wikivoyage article links.
    Phase 2 — Scraping:  fetch each discovered article, extract sections using
              the same parser as scrape_wiki.py, and write one record per section.

Output schema per JSONL record:
    source, url, itinerary_slug, section, category, text
    
Usage:
    python src/scrape_itineraries.py
    python src/scrape_itineraries.py --seeds Itineraries Itineraries_index --sleep 1.5
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from scrape_wiki import (
    BASE_URL,
    SECTION_CATEGORY,
    append_jsonl,
    extract_sections,
    fetch_page,
)

DEFAULT_SEED_PAGES = ("Itineraries", "Itineraries_index")
DEFAULT_OUTPUT     = Path("data/raw/wikivoyage_itineraries.jsonl")

META_TITLE_PREFIXES = ("Wikivoyage:", "Help:", "Template:", "Category:", "File:", "Special:", "MediaWiki:", "Module:", "User:")

def href_to_article_slug(href: str) -> Optional[str]:
    href = href.strip()
    if not href or href.startswith("#"):
        return None

    full   = urljoin("https://en.wikivoyage.org/", href)
    parsed = urlparse(full)

    if "wikivoyage.org" not in parsed.netloc.lower():
        return None

    if parsed.path.startswith("/wiki/"):
        slug = unquote(parsed.path[len("/wiki/"):].split("#")[0])
        return slug.replace(" ", "_") if slug else None

    if "title" in parse_qs(parsed.query):
        slug = unquote(parse_qs(parsed.query)["title"][0]).split("#")[0]
        return slug.replace(" ", "_") if slug else None

    return None

def is_meta_slug(slug: str) -> bool:    
    return any(slug.startswith(p) for p in META_TITLE_PREFIXES)

def extract_article_slugs_from_html(html: str) -> Set[str]:
    
    soup    = BeautifulSoup(html, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        return set()

    slugs: Set[str] = set()
    for anchor in content.find_all("a", href=True):
        slug = href_to_article_slug(anchor["href"])
        if slug and not is_meta_slug(slug):
            slugs.add(slug)
    return slugs

def collect_slugs_from_seed_pages(seed_slugs: List[str]) -> Set[str]:
    collected: Set[str] = set()
    for page_slug in seed_slugs:
        url  = f"{BASE_URL}/{page_slug}"
        html = fetch_page(url)
        if html is None:
            continue
        collected |= extract_article_slugs_from_html(html)
        collected.discard(page_slug)
    return collected

def scrape_itinerary_article(slug: str) -> List[Dict]:
    url      = f"{BASE_URL}/{slug}"
    html     = fetch_page(url)
    if html is None:
        return []

    sections = extract_sections(html)
    records: List[Dict] = []

    for section_name, section_text in sections.items():
        if len(section_text) < 50:
            continue
        records.append({
            "source":          "wikivoyage",
            "url":             url,
            "itinerary_slug":  slug,
            "section":         section_name,
            "category":        SECTION_CATEGORY.get(section_name, "other"),
            "text":            section_text,
        })
    return records

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Collect itinerary article links from Wikivoyage hub pages, then scrape sections.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=list(DEFAULT_SEED_PAGES),
        help="Wikivoyage article slugs to scan for outbound links (default: Itineraries Itineraries_index).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="JSONL output path.",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to sleep between article requests.",
    )
    args = parser.parse_args()

    slugs = sorted(collect_slugs_from_seed_pages(args.seeds))
    if not slugs:
        print("No article slugs found from seed pages.")
        return

    if args.output.exists():
        args.output.unlink()

    total_records = 0
    for slug in tqdm(slugs, desc="Itinerary articles"):
        records = scrape_itinerary_article(slug)
        append_jsonl(records, args.output)
        total_records += len(records)
        time.sleep(args.sleep)

    print(f"Done. Saved {total_records} records for {len(slugs)} articles to {args.output}")

if __name__ == "__main__":
    main()