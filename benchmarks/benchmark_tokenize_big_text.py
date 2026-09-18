# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers", "requests"]
# ///
"""Benchmark HuggingFace's tokenizers library on big.txt.

Usage: uv run benchmarks/benchmark_tokenize_big_text.py [iterations] [model]
"""

import os
import sys
import time
from pathlib import Path

import requests
from tokenizers import Tokenizer


def cache_dir() -> Path:
    env = os.environ.get("GO_POTION_HOME")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "go-potion"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "go-potion"


def big_text_chunks() -> list[str]:
    path = cache_dir() / "big.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get("https://norvig.com/big.txt", timeout=60).content)
    words = path.read_text(encoding="utf-8").split(" ")
    return [" ".join(words[i : i + 256]) for i in range(0, len(words), 256)]


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = sys.argv[2] if len(sys.argv) > 2 else "lightonai/mLateOn"

    start = time.perf_counter()
    tokenizer = Tokenizer.from_pretrained(model)
    tokenizer.no_truncation()
    print(f"Loaded {model} in {time.perf_counter() - start:.2f}s")

    chunks = big_text_chunks()
    total_bytes = sum(len(chunk.encode("utf-8")) for chunk in chunks)

    elapsed = 0.0
    total_tokens = 0
    for _ in range(iterations):
        start = time.perf_counter()
        for chunk in chunks:
            total_tokens += len(tokenizer.encode(chunk, add_special_tokens=False).ids)
        elapsed += time.perf_counter() - start

    mb = total_bytes * iterations / 1024 / 1024
    print(f"{mb / elapsed:.2f} MB/s  {total_tokens / elapsed / 1e6:.2f} Mtokens/s  ({iterations} iterations, {len(chunks)} chunks)")


if __name__ == "__main__":
    main()
