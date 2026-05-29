"""
infer.py - Autoregressive text generation from a trained transformer checkpoint.

Exposes a single public function, generate(), used by app.py (Gradio UI) and
callable from the command line.  The vectorizer and model are initialised once
at module import time so repeated calls from the UI are fast.

Sampling strategy (applied in order):
    1. Temperature scaling  — controls distribution sharpness.
    2. Nucleus (top-p)      — zeroes low-probability tail before sampling.
    3. Repetition penalty   — discourages tokens already in the generated output.

Token ID conventions (set by Keras TextVectorization):
    0  = padding  →  treated as end-of-sequence; generation stops on this token.
    1  = [UNK]    →  generation continues past UNK but the token is stripped from
                     the returned string.

                     Usage (CLI):
    python src/infer.py --prompt "Country: France\\nCity: Paris\\nCategory: Eat\\nSection: Restaurants\\nTravel information:"
    python src/infer.py --prompt "..." --max-tokens 120 --temperature 0.7
"""

# Suppress startup noise
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformer import (
    create_model,
    VOCAB_SIZE,
    MAXLEN,
    EMBED_DIM,
    NUM_HEADS,
    FEED_FORWARD_DIM,
)


SEQ_LEN = MAXLEN - 1

vocab = Path("artifacts/tokenizer/vocabulary.txt").read_text(encoding="utf-8").splitlines()

vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAXLEN, 
    standardize="lower_and_strip_punctuation",
    split="whitespace",
)
vectorizer.set_vocabulary(vocab)

model = create_model(
    maxlen=SEQ_LEN,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FEED_FORWARD_DIM,
    num_layers=2,
)
model.load_weights("checkpoints/best.weights.h5")

#Generate text autoregressively from a prompt
def generate(
    prompt_text: str,
    max_new_tokens: int = 80,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> str:
    
    vocab_list = vectorizer.get_vocabulary()

    # Tokenize the prompt to MAXLEN integer ID
    prompt_tensor = vectorizer(tf.expand_dims(prompt_text, 0))
    all_tokens = prompt_tensor.numpy()[0].tolist()

    # Determine actual prompt length by finding the last non-zero token
    prompt_len = next((i for i in range(len(all_tokens) - 1, -1, -1) if all_tokens[i] != 0), 0) + 1
    context = all_tokens[:min(prompt_len, SEQ_LEN)]

    new_tokens: list[int] = []

    for _ in range(max_new_tokens):
        pad_len = SEQ_LEN - len(context)

        if pad_len > 0:            
            padded = context + [0] * pad_len
            predict_pos = len(context) - 1
        else:            
            padded = context[-SEQ_LEN:]
            predict_pos = SEQ_LEN - 1

        x = np.array([padded], dtype=np.int32)
        logits = model(x, training=False).numpy()
        next_logits = logits[0, predict_pos].copy()
        
        if repetition_penalty != 1.0:
            for token_id in set(new_tokens):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        if temperature <= 0:
            next_id = int(np.argmax(next_logits))
        else:            
            scaled = next_logits / temperature
            scaled -= scaled.max() 
            probs = np.exp(scaled)
            probs /= probs.sum()
            
            if top_p < 1.0:
                sorted_idx  = np.argsort(probs)[::-1]
                cumulative  = np.cumsum(probs[sorted_idx])
                cutoff_mask = cumulative > top_p
                if cutoff_mask.any():
                    first_over = int(np.argmax(cutoff_mask))                    
                    probs[sorted_idx[first_over + 1:]] = 0.0
                    probs /= probs.sum()

            next_id = int(np.random.choice(len(probs), p=probs))

        if next_id == 0:
            break

        new_tokens.append(next_id)
        context.append(next_id)        

    words = [vocab_list[i] for i in new_tokens if 1 < i < len(vocab_list)]
    return " ".join(words)

def main() -> None:
    """Run a single generation from the command line and print the result."""
    parser = argparse.ArgumentParser(description="Generate travel text from a prompt.")
    parser.add_argument("--prompt",      type=str,   required=True, help="Seed prompt text (use \\n for newlines).")
    parser.add_argument("--max-tokens",  type=int,   default=80,    help="Maximum new tokens to generate (default: 80).")
    parser.add_argument("--temperature", type=float, default=0.8,   help="Sampling temperature (default: 0.8).")
    args = parser.parse_args()
    
    prompt = args.prompt.replace("\\n", "\n")
    print(generate(prompt, args.max_tokens, args.temperature))

if __name__ == "__main__":
    main()