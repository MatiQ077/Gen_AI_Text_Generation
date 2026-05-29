"""
generate_synthetic.py — Generate synthetic travel itineraries via the Claude API.

Calls claude-haiku for each (destination, trip_length) combination to produce
itineraries in the exact training prompt format.  The system prompt is sent
with ephemeral cache_control so it is prompt-cached across all ~200 API calls,
significantly reducing token cost.

Destinations: ~100 cities across all continents × 2 trip lengths (3 and 5 days)
= ~200 API calls total.

The resulting JSONL is consumed by src_gpt/train.py as SYNTHETIC_CORPUS, where
it is oversampled (×5) to keep the structured Day/Morning/Afternoon/Evening
format well-represented in training.

Output: data/processed/synthetic_itineraries.jsonl

Usage:
    python src_gpt/generate_synthetic.py
    python src_gpt/generate_synthetic.py --out data/processed/synthetic_itineraries.jsonl
    python src_gpt/generate_synthetic.py --dry-run   # print first prompt, no API call

"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = Path("data/processed/synthetic_itineraries.jsonl")

SYSTEM_PROMPT = """\
You generate travel itineraries in a strict format. Output ONLY this structure, no extra text:

Country: [country]
City: [city]
Category: itinerary
Section: [N] day trip
Travel information: Day 1:
Morning: [2-3 specific activities with real place names]
Afternoon: [2-3 activities with real place names]
Evening: [dinner recommendation at a real restaurant or area + 1 evening activity]
Day 2:
Morning: ...
Afternoon: ...
Evening: ...
[continue for all days]"""

DESTINATIONS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Barcelona"),
    ("United Kingdom", "London"), ("United States", "New York"), ("United States", "San Francisco"),
    ("Australia", "Sydney"), ("Germany", "Berlin"), ("Netherlands", "Amsterdam"),
    ("Portugal", "Lisbon"), ("Greece", "Athens"), ("Turkey", "Istanbul"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Vietnam", "Ho Chi Minh City"), ("Indonesia", "Bali"),
    ("Singapore", "Singapore"), ("India", "Mumbai"), ("India", "Delhi"),
    ("India", "Jaipur"), ("China", "Beijing"), ("China", "Shanghai"),
    ("South Korea", "Seoul"), ("Hong Kong", "Hong Kong"), ("Taiwan", "Taipei"),
    ("Morocco", "Marrakech"), ("Morocco", "Fez"), ("Egypt", "Cairo"),
    ("South Africa", "Cape Town"), ("Kenya", "Nairobi"), ("Tanzania", "Zanzibar"),
    ("Mexico", "Mexico City"), ("Mexico", "Oaxaca"), ("Mexico", "Cancun"),
    ("Peru", "Lima"), ("Peru", "Cusco"), ("Brazil", "Rio de Janeiro"),
    ("Brazil", "Sao Paulo"), ("Argentina", "Buenos Aires"), ("Colombia", "Cartagena"),
    ("Cuba", "Havana"), ("Canada", "Vancouver"), ("Canada", "Montreal"),
    ("United States", "New Orleans"), ("United States", "Chicago"), ("United States", "Miami"),
    ("Austria", "Vienna"), ("Czech Republic", "Prague"), ("Hungary", "Budapest"),
    ("Poland", "Krakow"), ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"),
    ("Norway", "Bergen"), ("Finland", "Helsinki"), ("Switzerland", "Zurich"),
    ("Belgium", "Brussels"), ("Ireland", "Dublin"), ("Scotland", "Edinburgh"),
    ("Croatia", "Dubrovnik"), ("Iceland", "Reykjavik"), ("Israel", "Jerusalem"),
    ("Jordan", "Petra"), ("UAE", "Dubai"), ("Qatar", "Doha"),
    ("Malaysia", "Kuala Lumpur"), ("Philippines", "Manila"), ("Sri Lanka", "Colombo"),
    ("Nepal", "Kathmandu"), ("Bhutan", "Thimphu"), ("Myanmar", "Yangon"),
    ("Cambodia", "Siem Reap"), ("Laos", "Luang Prabang"), ("New Zealand", "Auckland"),
    ("New Zealand", "Queenstown"), ("Italy", "Florence"), ("Italy", "Venice"),
    ("Spain", "Madrid"), ("Spain", "Seville"), ("France", "Nice"),
    ("France", "Lyon"), ("Germany", "Munich"), ("Japan", "Kyoto"),
    ("Japan", "Osaka"), ("United States", "Los Angeles"), ("United States", "Washington DC"),
    ("United States", "Boston"), ("Canada", "Toronto"), ("Mexico", "Guadalajara"),
    ("Chile", "Santiago"), ("Ecuador", "Quito"), ("Bolivia", "La Paz"),
    ("Rwanda", "Kigali"), ("Ethiopia", "Addis Ababa"), ("Ghana", "Accra"),
    ("Senegal", "Dakar"), ("Tunisia", "Tunis"), ("South Africa", "Johannesburg"),
    ("Maldives", "Male"), ("Seychelles", "Victoria"), ("Mauritius", "Port Louis"),
    ("Greece", "Santorini"), ("Spain", "Mallorca"), ("Italy", "Amalfi"),
]

TRIP_LENGTHS = [3, 5]

def make_user_prompt(city: str, country: str, days: int) -> str:
    return (
        f"Generate a {days}-day itinerary for {city}, {country}. "
        f"Include real, specific place names, restaurants, and neighborhoods."
    )

def parse_response(text: str, city: str, country: str, days: int) -> dict | None:
    
    text = text.strip()
    if not text.startswith("Country:"):
        return None
    travel_text = text.split("Travel information:", 1)[1].strip() if "Travel information:" in text else text
    return {
        "source":        "synthetic",
        "country":       country,
        "city":          city,
        "category":      "itinerary",
        "section":       f"{days} day trip",
        "text":          travel_text,
        "training_text": text,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first prompt and exit without calling API.",
    )
    args = parser.parse_args()

    if args.dry_run:
        country, city = DESTINATIONS[0]
        print(make_user_prompt(city, country, 3))
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    total   = len(DESTINATIONS) * len(TRIP_LENGTHS)
    skipped = 0

    for i, (country, city) in enumerate(DESTINATIONS):
        for days in TRIP_LENGTHS:
            idx         = i * len(TRIP_LENGTHS) + TRIP_LENGTHS.index(days) + 1
            user_prompt = make_user_prompt(city, country, days)
            print(f"[{idx}/{total}] {city}, {country} — {days} days... ", end="", flush=True)

            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=700,
                    system=[{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}, 
                    }],
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text   = response.content[0].text
                record = parse_response(text, city, country, days)
                if record:
                    records.append(record)
                    print("ok")
                else:
                    skipped += 1
                    print("SKIPPED (unexpected format)")
            except Exception as e:
                skipped += 1
                print(f"ERROR: {e}")

            time.sleep(0.3)

    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved {len(records)} records to {args.out} ({skipped} skipped).")

if __name__ == "__main__":
    main()