"""Precompute and cache FLUX.2 klein text embeddings, one file per distinct prompt.

Runs the text encoder (Qwen3) completely on its own -- no transformer, no VAE loaded
at all -- so it uses only the text encoder's own VRAM footprint, then exits and frees
it entirely. train.sh's training script checks this cache before encoding anything,
so with the empty/fixed prompt every example in this project uses, the text encoder
never needs to touch the GPU during actual training: run this once, then train.

Every prompt gets its own cache file (sha256(prompt + max_sequence_length).pt), so a
prompt already cached from a previous run -- or from a different box's dataset, since
the prompt is often shared -- is skipped rather than re-encoded.

Usage:
    python src2/precompute_text_embeds.py --prompt "" --cache-dir src2/text_embed_cache
    python src2/precompute_text_embeds.py --prompts-file prompts.txt --cache-dir src2/text_embed_cache
"""

import argparse
import hashlib
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline


def cache_key(prompt: str, max_sequence_length: int, cache_dir: Path) -> Path:
    h = hashlib.sha256(f"{prompt}|{max_sequence_length}".encode()).hexdigest()
    return cache_dir / f"{h}.pt"


def precompute(prompts: list[str], model_id: str, max_sequence_length: int, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in prompts if not cache_key(p, max_sequence_length, cache_dir).exists()]
    if not todo:
        print(f"all {len(prompts)} prompt(s) already cached in {cache_dir}")
        return

    print(f"encoding {len(todo)}/{len(prompts)} prompt(s) not yet cached")
    # vae=None, transformer=None: only the tokenizer + Qwen3 text encoder load, so this
    # process's peak VRAM is just the text encoder's, never co-resident with the transformer.
    pipe = Flux2KleinPipeline.from_pretrained(model_id, vae=None, transformer=None, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for prompt in todo:
            prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompt, max_sequence_length=max_sequence_length)
            path = cache_key(prompt, max_sequence_length, cache_dir)
            torch.save({"prompt_embeds": prompt_embeds.cpu(), "text_ids": text_ids.cpu()}, path)
            print(f"cached {prompt!r} -> {path.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompt", action="append", default=[], help="a prompt to cache; repeatable")
    p.add_argument("--prompts-file", type=Path, default=None, help="file with one prompt per line")
    p.add_argument("--max-sequence-length", type=int, default=8)
    p.add_argument("--cache-dir", type=Path, required=True)
    args = p.parse_args()

    prompts = list(args.prompt)
    if args.prompts_file:
        prompts += [line for line in args.prompts_file.read_text().splitlines()]
    if not prompts:
        prompts = [""]  # this project's default: a single fixed empty prompt

    precompute(prompts, args.model_id, args.max_sequence_length, args.cache_dir)
