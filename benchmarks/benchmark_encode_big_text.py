#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["model2vec", "requests"]
# ///
"""
Benchmark script to test encoding big text using the Potion model.
This replicates the functionality of BenchmarkEncodeBigText in the Go implementation.
"""

import os
import sys
import time
import requests
from pathlib import Path
from model2vec import StaticModel

def go_potion_cache_dir() -> str:
    """Mirror go-potion's cache resolution: GO_POTION_HOME, then the
    platform user cache directory."""
    env = os.environ.get("GO_POTION_HOME")
    if env:
        return env
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/go-potion")
    if sys.platform == "win32":
        return os.path.join(os.environ["LOCALAPPDATA"], "go-potion")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return os.path.join(xdg, "go-potion")
    return os.path.expanduser("~/.cache/go-potion")

def download_big_text(cache_dir: str) -> str:
    """Download big.txt if it doesn't exist."""
    big_text_path = os.path.join(cache_dir, "big.txt")
    
    if not os.path.exists(big_text_path):
        print("Downloading big.txt...")
        response = requests.get("https://norvig.com/big.txt")
        response.raise_for_status()
        
        with open(big_text_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Downloaded big.txt successfully")
    
    return big_text_path

def benchmark_encode_big_text(num_iterations: int = 1, model_id: str = "minishlab/potion-base-2M"):
    """Benchmark encoding big text similar to Go's BenchmarkEncodeBigText."""

    # Share big.txt with the Go and Rust benchmarks via go-potion's cache
    cache_dir = go_potion_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    # Create the model
    print(f"Loading {model_id}...")
    model = StaticModel.from_pretrained(model_id)
    print("Model loaded successfully")
    
    # Download big.txt if it doesn't exist
    big_text_path = download_big_text(cache_dir)
    
    # Read the file
    print("Reading big.txt...")
    with open(big_text_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split into words
    words = content.split()
    print(f"Total words: {len(words)}")
    
    # Process in chunks of 256 words
    chunk_size = 256
    total_bytes = 0
    total_vectors = 0
    total_time = 0.0
    
    print(f"Starting benchmark with {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        start_time = time.time()
        
        for j in range(0, len(words), chunk_size):
            end = j + chunk_size
            if end > len(words):
                end = len(words)
            
            chunk = " ".join(words[j:end])
            
            # Encode the chunk
            embedding = model.encode(chunk)
            
            total_bytes += len(chunk.encode('utf-8'))
            total_vectors += 1
        
        iteration_time = time.time() - start_time
        total_time += iteration_time
        
        # Calculate metrics for this iteration
        mb_per_second = (total_bytes / iteration_time) / 1024 / 1024
        vectors_per_second = total_vectors / iteration_time
        
        print(f"  Iteration {iteration + 1} results:")
        print(f"    Time: {iteration_time:.2f}s")
        print(f"    MB/s: {mb_per_second:.2f}")
        print(f"    Vectors/s: {vectors_per_second:.2f}")
        print(f"    Total chunks processed: {total_vectors}")
        print(f"    Total bytes processed: {total_bytes:,}")
    
    # Calculate overall metrics
    avg_mb_per_second = (total_bytes / total_time) / 1024 / 1024
    avg_vectors_per_second = total_vectors / total_time
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Total iterations: {num_iterations}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average MB/s: {avg_mb_per_second:.2f}")
    print(f"Average vectors/s: {avg_vectors_per_second:.2f}")
    print(f"Total chunks processed: {total_vectors}")
    print(f"Total bytes processed: {total_bytes:,}")
    print("="*50)

if __name__ == "__main__":
    import sys
    
    # Default to 1 iteration of potion-base-2M, but allow command line arguments
    num_iterations = 1
    model_id = "minishlab/potion-base-2M"
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print("Usage: python benchmark_encode_big_text.py [num_iterations] [model_id]")
            sys.exit(1)
    if len(sys.argv) > 2:
        model_id = sys.argv[2]

    benchmark_encode_big_text(num_iterations, model_id) 