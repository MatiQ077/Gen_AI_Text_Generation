"""
scrape_wiki.py — Scrape city guide pages from Wikivoyage and save as JSONL.

Reads a CSV of cities (country, city, wikivoyage_slug), fetches the Wikivoyage
page for each city, parses known section headings, and appends one record per
section to the output JSONL file.

A "slug" is the URL-safe identifier after /wiki/ in a Wikivoyage URL

Output schema per JSONL record:
    source, url, country, city, section, category, text

This module also exports BASE_URL, SECTION_CATEGORY, fetch_page,
extract_sections, and append_jsonl for use by scrape_itineraries.py.

Usage:
    python src/scrape_wiki.py
    python src/scrape_wiki.py --city-list data/my_cities.csv --sleep 1.5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL       = "https://en.wikivoyage.org/wiki"
OUTPUT_PATH    = Path("data/raw/wikivoyage_pages.jsonl")
CITY_LIST_PATH = Path("data/city_list_500.csv")

SECTION_CATEGORY = {
    "Understand":  "city_description",
    "Get in":      "transportation",
    "Get around":  "transportation",
    "See":         "attractions",
    "Do":          "activities",
    "Buy":         "shopping",
    "Eat":         "food",
    "Drink":       "nightlife",
    "Sleep":       "accommodation",
    "Stay safe":   "safety",
    "Connect":     "practical_info",
}

def fetch_page(url: str, timeout: int = 20) -> Optional[str]:
    
    headers = {"User-Agent": "TravelAssistantStudentProject"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as error:
        print(f"[ERROR] Could not fetch {url}: {error}")
        return None

def basic_clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

def extract_sections(html: str) -> Dict[str, str]:
    
    soup = BeautifulSoup(html, "lxml")
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        return {}

    sections: Dict[str, str] = {}
    current_section = None
    collected_text: List[str] = []

    for element in content.find_all(["h2", "h3", "p", "ul"], recursive=True):
        if element.name in ["h2", "h3"]:            
            if current_section and collected_text:
                section_text = basic_clean(" ".join(collected_text))
                if section_text:
                    sections[current_section] = section_text

            headline = element.get_text(" ", strip=True).replace("[edit]", "").strip()
            current_section = None
            collected_text = []

            for known_section in SECTION_CATEGORY:
                if headline.lower().startswith(known_section.lower()):
                    current_section = known_section
                    break

        elif current_section:
            paragraph_text = basic_clean(element.get_text(" ", strip=True))
            if paragraph_text:
                collected_text.append(paragraph_text)

    if current_section and collected_text:
        section_text = basic_clean(" ".join(collected_text))
        if section_text:
            sections[current_section] = section_text

    return sections

def scrape_city(country: str, city: str, slug: str) -> List[Dict]:
   
    url  = f"{BASE_URL}/{slug}"
    html = fetch_page(url)
    if html is None:
        return []

    sections = extract_sections(html)
    records: List[Dict] = []

    for section_name, section_text in sections.items():
        if len(section_text) < 50:
            continue
        records.append({
            "source":   "wikivoyage",
            "url":      url,
            "country":  country,
            "city":     city,
            "section":  section_name,
            "category": SECTION_CATEGORY.get(section_name, "other"),
            "text":     section_text,
        })

    return records

def append_jsonl(records: List[Dict], output_path: Path) -> None:
   
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:    
    parser = argparse.ArgumentParser(description="Scrape city pages from Wikivoyage.")
    parser.add_argument(
        "--city-list", type=Path, default=CITY_LIST_PATH,
        help="CSV with columns: country, city, wikivoyage_slug (default: city_list_500.csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="JSONL output path. Existing file is overwritten.",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds between requests (default: 1.0).",
    )
    args = parser.parse_args()

    cities_df = pd.read_csv(args.city_list)

    if args.output.exists():
        args.output.unlink()

    total_records = 0

    for _, row in tqdm(cities_df.iterrows(), total=len(cities_df)):
        records = scrape_city(row["country"], row["city"], row["wikivoyage_slug"])
        append_jsonl(records, args.output)
        total_records += len(records)
        time.sleep(args.sleep)

    print(f"Done. Saved {total_records} records to {args.output}")

if __name__ == "__main__":
    main()