import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from transformer import (
    EMBED_DIM,
    FEED_FORWARD_DIM,
    MAXLEN,
    NUM_HEADS,
    VOCAB_SIZE,
    create_model,
)

TRAVEL_CORPUS = Path("data/processed/travel_corpus.jsonl")
ITINERARY_CORPUS = Path("data/processed/itinerary_corpus.jsonl")
PDF_CORPUS = Path("data/processed/pdf_corpus.jsonl")
VAL_FRACTION = 0.05
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 5e-4
DROPOUT_RATE = 0.3
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_DIR = Path("artifacts/tokenizer")
VOCAB_PATH = TOKENIZER_DIR / "vocabulary.txt"
RANDOM_SEED = 42

def record_to_lm_string(rec: dict) -> str:
    if rec.get("training_text"):
        return rec["training_text"]
    return (
        f"Itinerary: {rec.get('itinerary_slug', '')}\n"
        f"Section: {rec.get('section', '')}\n"
        f"Category: {rec.get('category', '')}\n"
        f"Travel information: {rec['text']}"
    )

def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_corpus_strings() -> List[str]:
    strings: List[str] = []
    for path in (TRAVEL_CORPUS, ITINERARY_CORPUS, PDF_CORPUS):
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        for rec in load_jsonl(path):
            text = record_to_lm_string(rec).strip()
            if text:
                strings.append(text)
    return strings

def train_val_split(
    strings: List[str], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = strings.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction))
    val_strings = shuffled[:val_size]
    train_strings = shuffled[val_size:]
    return train_strings, val_strings

def build_vectorizer(train_strings: List[str]) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAXLEN,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(train_strings)
    return vectorizer

def save_vocabulary(vectorizer: layers.TextVectorization, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab = vectorizer.get_vocabulary()
    path.write_text("\n".join(vocab), encoding="utf-8")

def make_lm_dataset(
    strings: List[str],
    vectorizer: layers.TextVectorization,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    def to_xy(batch_strings: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        tokens = vectorizer(batch_strings)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(strings)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(strings), 10_000), seed=RANDOM_SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(to_xy, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def masked_sparse_ce(y_true, y_pred):
    """Cross-entropy ignoring padded positions (label id 0)."""
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    per_token = loss_fn(y_true, y_pred)
    weighted = per_token * mask
    return tf.reduce_sum(weighted) / tf.maximum(tf.reduce_sum(mask), 1.0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train decoder-only travel LM.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train on 32 samples for 3 epochs (sanity check).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading corpora...")
    strings = load_corpus_strings()
    print(f"Total examples: {len(strings)}")

    train_strings, val_strings = train_val_split(strings, VAL_FRACTION, RANDOM_SEED)
    print(f"Train: {len(train_strings)}, Val: {len(val_strings)}")

    if args.smoke:
        train_strings = train_strings[:32]
        val_strings = val_strings[:8]
        args.epochs = 3
        print("Smoke mode: using small subset.")

    print("Building tokenizer...")
    vectorizer = build_vectorizer(train_strings)
    save_vocabulary(vectorizer, VOCAB_PATH)
    print(f"Saved vocabulary ({len(vectorizer.get_vocabulary())} tokens) to {VOCAB_PATH}")

    train_ds = make_lm_dataset(
        train_strings, vectorizer, args.batch_size, shuffle=True
    )
    val_ds = make_lm_dataset(
        val_strings, vectorizer, args.batch_size, shuffle=False
    )

    seq_len = MAXLEN - 1
    print(f"Sequence length (tokens): {seq_len}, MAXLEN: {MAXLEN}")

    model = create_model(
        maxlen=seq_len,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FEED_FORWARD_DIM,
        num_layers=2,
        dropout=DROPOUT_RATE,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
        loss=masked_sparse_ce,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "best.weights.h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save_weights(checkpoint_path)
    print(f"Best weights saved to {checkpoint_path}")
    print(f"Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"Final val loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()