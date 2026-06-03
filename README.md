# RAG-Enhanced Travel Itinerary Generator

Enter a destination and trip preferences → get a structured day-by-day itinerary grounded in a travel knowledge base.

## Background

The project started by building a **decoder-only Transformer from scratch** (Keras/TensorFlow, `src/`) trained on travel text scraped from Wikivoyage, Reddit, StackExchange, and PDF guides. After training, it became clear there was not enough high-quality, structured itinerary data to produce coherent multi-day itineraries — the model learned travel vocabulary but couldn't sustain day-by-day structure.

The project pivoted to **fine-tuning GPT-2** (PyTorch/Hugging Face, `src_gpt/`) instead, which leverages a strong pretrained base and needs far less task-specific data to generate well-structured output. To further improve factual grounding, a **RAG pipeline** (`src/rag.py`) retrieves relevant travel chunks from a FAISS index and injects them into the prompt before generation.

The active inference backend is GPT-2. The from-scratch Transformer code is preserved in `src/` and remains runnable.

## Architecture

```
Data sources (Wikivoyage, Reddit, StackExchange, PDFs, Synthetic)
        │
        ▼
  Data pipeline (scraping → clean_data.py → FAISS index)
        │
        ▼
  FAISS knowledge base  ◄──── User input (destination + preferences)
        │                              │
        └──────── RAG retrieval ───────┘
                        │
                        ▼
              GPT-2 (fine-tuned, src_gpt/)
                        │
                        ▼
                  Gradio web UI (app.py)
```

## Project Structure

```
├── app.py                   # Gradio UI entry point
├── src/                     # Original from-scratch pipeline
│   ├── transformer.py       # Decoder-only Transformer (Keras)
│   ├── train.py             # Train custom Transformer
│   ├── infer.py             # Inference with custom Transformer
│   ├── rag.py               # RAG pipeline (shared by both backends)
│   ├── build_index.py       # Build FAISS retrieval index
│   ├── clean_data.py        # Raw → processed corpus conversion
│   ├── scrape_wiki.py       # Scrape Wikivoyage city pages
│   ├── scrape_itineraries.py
│   ├── scrape_reddit.py
│   ├── scrape_stackexchange.py
│   └── get_pdf_data.py
├── src_gpt/                 # Active GPT-2 fine-tuning pipeline
│   ├── train.py             # Fine-tune GPT-2
│   ├── infer.py             # Inference with fine-tuned GPT-2
│   └── generate_synthetic.py  # Generate synthetic data via Claude API
├── data/
│   ├── raw/                 # Raw scraped JSONL files (gitignored)
│   └── processed/           # Cleaned, chunked JSONL files (gitignored)
└── artifacts/
    ├── gpt2_finetuned/      # Saved fine-tuned GPT-2 checkpoint
    ├── rag_index.faiss      # FAISS retrieval index
    └── tokenizer/           # Vocabulary for custom Transformer
```

## Setup

Requires **Python 3.10+**. From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux use `source .venv/bin/activate` instead of `.venv\Scripts\activate`.

## Run Order

**1. Collect data**

```bash
python src/scrape_wiki.py          # Wikivoyage city guides
python src/scrape_itineraries.py   # Wikivoyage itinerary routes
python src/scrape_reddit.py        # Reddit trip reports
python src/scrape_stackexchange.py # Travel StackExchange Q&A
python src/get_pdf_data.py         # PDF travel guides
```

**2. Process data**

```bash
python src/clean_data.py
```

**3. Generate synthetic itineraries** *(optional — requires Claude API key in `.env`)*

```bash
python src_gpt/generate_synthetic.py
```

**4. Fine-tune GPT-2**

```bash
python src_gpt/train.py
```

**5. Build FAISS index**

```bash
python src/build_index.py
```

**6. Launch the app**

```bash
python app.py
```

## Custom Transformer (Legacy)

The original from-scratch Transformer lives in `src/`. To switch back to it, change one import line in `src/rag.py`:

```python
# Active (GPT-2):
from src_gpt.infer import generate

# Legacy (custom Transformer):
from src.infer import generate
```

For a smoke-test of the custom Transformer training loop:

```bash
python src/train.py --smoke
```

## Data Sources

| Source | Script | Notes |
|---|---|---|
| Wikivoyage city guides | `scrape_wiki.py` | Configure cities in `data/city_list.csv` |
| Wikivoyage itinerary routes | `scrape_itineraries.py` | Pre-built multi-day routes |
| Reddit trip reports | `scrape_reddit.py` | r/travel and related subreddits |
| Travel StackExchange | `scrape_stackexchange.py` | Q&A posts |
| PDF travel guides | `get_pdf_data.py` | Local PDFs |
| Synthetic itineraries | `generate_synthetic.py` | Claude API; oversampled ×5 in GPT-2 training |

Large `.jsonl` outputs under `data/raw/` and `data/processed/` are gitignored; regenerate them locally after cloning.

## License and data use

Scraped content comes from public Wikivoyage pages; comply with Wikimedia licensing and robots/community norms when crawling or redistributing derivatives.
