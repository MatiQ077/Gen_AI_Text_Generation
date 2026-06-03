"""
src_gpt/train.py — Fine-tune GPT-2 on itinerary-focused travel data (PyTorch).

Loads three curated corpora, applies quality and structure filters, tokenizes
to fixed-length blocks, and fine-tunes a GPT-2 base model with a cosine LR
schedule.  The best checkpoint (lowest validation loss) is saved to
artifacts/gpt2_finetuned/ in Hugging Face format.

Corpora and rationale:
    itinerary_corpus.jsonl  — Wikivoyage itinerary articles.
                              Filtered for Day N + time-of-day markers to
                              exclude narrative prose that would teach the
                              model to generate descriptions rather than
                              structured Day/Morning/Evening output.
    reddit_itineraries.jsonl — Reddit trip reports.
                              Already filtered for structure during scraping.
    synthetic_itineraries.jsonl — Claude-generated itineraries.
                              Oversampled ×SYNTHETIC_OVERSAMPLE to keep the
                              structured format signal strong in training.

General city text (travel_corpus, pdf_corpus, wikivoyage_extra, stackexchange)
is intentionally excluded because those sections contain transport listings and
narrative prose that degrade itinerary generation quality.

Quality filter (is_clean): rejects chunks shorter than 30 words, containing
URLs, phone numbers, listing symbols etc.

Usage:
    python src_gpt/train.py --smoke      # 50 train / 10 val, 2 epochs (sanity check)
    python src_gpt/train.py              # full training
    python src_gpt/train.py --epochs 10 --batch-size 8 --lr 2e-5
"""

import argparse
import json
import re
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_cosine_schedule_with_warmup

ITINERARY_CORPUS = Path("data/processed/itinerary_corpus.jsonl")
SYNTHETIC_CORPUS = Path("data/processed/synthetic_itineraries.jsonl")
REDDIT_CORPUS    = Path("data/raw/reddit_itineraries.jsonl")
SAVE_DIR = Path("artifacts/gpt2_finetuned")

BLOCK_SIZE          = 512
BATCH_SIZE          = 4
EPOCHS              = 5
LR                  = 3e-5
WARMUP_RATIO        = 0.06
MAX_GRAD_NORM       = 1.0
VAL_FRACTION        = 0.05
RANDOM_SEED         = 42
SYNTHETIC_OVERSAMPLE = 5    # repeat synthetic examples N times to boost format signal
                            
_URL_RE     = re.compile(r'https?://', re.IGNORECASE)
_PHONE_RE   = re.compile(r'\+\d[\d\s\-]{6,}')
_LISTING_RE = re.compile(r'[☏✆]')
_DAY_RE     = re.compile(r'\bday\s*[1-9]\d?\b', re.IGNORECASE)
_TIME_RE    = re.compile(r'\b(morning|afternoon|evening|night|lunch|dinner|breakfast)\b', re.IGNORECASE)

def is_clean(text: str) -> bool:
   
    if len(text.split()) < 30:
        return False
    if _URL_RE.search(text):
        return False
    if _PHONE_RE.search(text):
        return False
    if _LISTING_RE.search(text):
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.04:
        return False
    return True

def has_itinerary_structure(text: str) -> bool:    
    return bool(_DAY_RE.search(text) and _TIME_RE.search(text))

def load_jsonl(path: Path) -> List[dict]:   
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_corpus_strings() -> List[str]:
    
    strings: List[str] = []

    if ITINERARY_CORPUS.exists():
        before = len(strings)
        for rec in load_jsonl(ITINERARY_CORPUS):
            text = rec.get("training_text", "").strip()
            if text and is_clean(text) and has_itinerary_structure(text):
                strings.append(text)
        print(f"  Loaded {len(strings) - before} structured examples from {ITINERARY_CORPUS.name}")
    else:
        print(f"[WARN] Corpus not found, skipping: {ITINERARY_CORPUS}")

    if REDDIT_CORPUS.exists():
        before = len(strings)
        for rec in load_jsonl(REDDIT_CORPUS):
            text = rec.get("training_text", "").strip()
            if text and is_clean(text):
                strings.append(text)
        print(f"  Loaded {len(strings) - before} examples from {REDDIT_CORPUS.name}")
    else:
        print(f"[WARN] Corpus not found, skipping: {REDDIT_CORPUS}")

    if SYNTHETIC_CORPUS.exists():
        before   = len(strings)
        examples = []
        for rec in load_jsonl(SYNTHETIC_CORPUS):
            text = rec.get("training_text", "").strip()
            if text and is_clean(text):
                examples.append(text)
        strings.extend(examples * SYNTHETIC_OVERSAMPLE)
        print(f"  Loaded {len(strings) - before} examples from {SYNTHETIC_CORPUS.name} (×{SYNTHETIC_OVERSAMPLE} oversample)")
    else:
        print(f"[WARN] Corpus not found, skipping: {SYNTHETIC_CORPUS}")

    return strings

def train_val_split(
    strings: List[str], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Randomly shuffle and split strings into train and validation sets."""
    rng      = random.Random(seed)
    shuffled = strings.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_size:], shuffled[:val_size]

# Tokenize all strings to Block_SIZE and return as a stacked tensor
def tokenize_strings(strings: List[str], tokenizer: GPT2TokenizerFast) -> torch.Tensor:
    all_ids = []
    for text in strings:
        enc = tokenizer(
            text,
            max_length=BLOCK_SIZE,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        all_ids.append(enc["input_ids"])
    return torch.cat(all_ids, dim=0)

def run_epoch(model, loader, optimizer, scheduler, pad_id, device, train: bool) -> float:
    
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for (input_ids,) in tqdm(loader, desc="train" if train else "val", leave=False):
            input_ids = input_ids.to(device)
            labels    = input_ids.clone()
            labels[labels == pad_id] = -100  # mask padding from loss

            loss = model(input_ids=input_ids, labels=labels).loss

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

    return total_loss / len(loader)

def main() -> None:    
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on itinerary data.")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run on 50 train / 10 val samples for 2 epochs.",
    )
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading corpora...")
    strings = load_corpus_strings()
    print(f"Total examples after quality filter: {len(strings)}")

    if len(strings) == 0:
        raise RuntimeError("No training examples found. Check corpus paths.")

    train_strings, val_strings = train_val_split(strings, VAL_FRACTION, RANDOM_SEED)
    print(f"Train: {len(train_strings)}, Val: {len(val_strings)}")

    if args.smoke:
        train_strings = train_strings[:50]
        val_strings   = val_strings[:10]
        args.epochs   = 2
        print("Smoke mode: 50 train / 10 val samples, 2 epochs.")

    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    print("Tokenizing...")
    train_ids = tokenize_strings(train_strings, tokenizer)
    val_ids   = tokenize_strings(val_strings,   tokenizer)
    print(f"Train tensor: {train_ids.shape}, Val tensor: {val_ids.shape}")

    train_loader = DataLoader(TensorDataset(train_ids), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_ids),   batch_size=args.batch_size, shuffle=False)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print("Loading GPT-2 model...")
    model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Scheduler: cosine warmup over {warmup_steps} steps, total {total_steps} steps.")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print("Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, pad_id, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, scheduler, pad_id, device, train=False)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(SAVE_DIR))
            tokenizer.save_pretrained(str(SAVE_DIR))
            print(f"  Saved best model (val_loss={best_val_loss:.4f}) → {SAVE_DIR}")

    print("Done.")

if __name__ == "__main__":
    main()