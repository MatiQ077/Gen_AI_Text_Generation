"""
src_gpt/infer.py — Inference backend using a fine-tuned GPT-2 model.

Loads the fine-tuned GPT-2 checkpoint from artifacts/gpt2_finetuned/ at
import time and exposes a generate() function with the same signature as
src/infer.py so rag.py can switch between the two backends by changing a
single import line.

Unlike src/infer.py (which implements sampling manually), this module delegates
all sampling to Hugging Face's model.generate(), which handles temperature,
top-p, and repetition penalty internally.

GPT-2 note: GPT-2 has no dedicated padding token.  The tokenizer's pad_token
is set to eos_token so that padding and sequence termination share the same ID.
The model is loaded in eval mode — call generate() only, never model.train().

Usage (CLI):
    python src_gpt/infer.py --prompt "Country: France\\nCity: Paris\\nCategory: itinerary\\nSection: 3 day trip\\nTravel information: Day 1:\\nMorning:"
    python src_gpt/infer.py --prompt "..." --max-tokens 150 --temperature 0.7
"""

import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_MODEL_DIR = Path("artifacts/gpt2_finetuned")
_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = GPT2TokenizerFast.from_pretrained(str(_MODEL_DIR))
_tokenizer.pad_token = _tokenizer.eos_token  # GPT-2 has no default pad token

_model = GPT2LMHeadModel.from_pretrained(str(_MODEL_DIR))
_model.to(_device)
_model.eval()

def generate(
    prompt_text: str,
    max_new_tokens: int = 80,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> str:
    
    max_prompt_len = 1024 - max_new_tokens
    enc = _tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
    )
    input_ids      = enc["input_ids"].to(_device)
    attention_mask = enc["attention_mask"].to(_device)
    prompt_len     = input_ids.shape[1]

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][prompt_len:]
    return _tokenizer.decode(new_ids, skip_special_tokens=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate travel text with fine-tuned GPT-2.")
    parser.add_argument("--prompt",      required=True,            help="Seed prompt text (use \\n for newlines).")
    parser.add_argument("--max-tokens",  type=int,   default=80,   help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,  help="Sampling temperature.")
    args = parser.parse_args()

    prompt = args.prompt.replace("\\n", "\n")
    print(generate(prompt, args.max_tokens, args.temperature))

if __name__ == "__main__":
    main()