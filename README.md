# RAG-Enhanced Text Generation for Personalized Travel Itineraries

Main goal is to build a **small decoder-only Transformer from scratch** trained on travel text, then extend it with **retrieval-augmented generation (RAG)** so itinerary outputs stay better grounded in a travel knowledge base.

## Setup

Requires **Python 3.10+** (recommended). From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux use `source .venv/bin/activate` instead of `.venv\Scripts\activate`.

## Data pipeline

1. Ensure `data/city_list.csv` lists the cities you want with columns **`country`**, **`city`**, and **`wikivoyage_slug`** (English Wikivoyage article title, e.g. `Paris` for https://en.wikivoyage.org/wiki/Paris).
2. Run scraping (polite `User-Agent` is set; respect Wikivoyage [terms of use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)):

   ```bash
   python src/scrape_wiki.py
   ```

3. Build processed chunks:

   ```bash
   python src/clean_data.py
   ```

Large `.jsonl` outputs under `data/raw/` and `data/processed/` are listed in `.gitignore`; regenerate them locally after clone.

## License and data use

Scraped content comes from public Wikivoyage pages; comply with Wikimedia licensing and robots/community norms when crawling or redistributing derivatives.