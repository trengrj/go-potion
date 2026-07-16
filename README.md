<p align="center">
  <img src="logo.png" alt="go-potion logo" width="200">
</p>

# go-potion

Static embeddings are lightweight, extremely fast, and require no GPU resources. They are ideal for high-throughput semantic search, classification, and retrieval where transformer-quality embeddings aren't required.

`go-potion` is a library for fast static text embedding inference in Go, supporting the [potion](https://huggingface.co/collections/minishlab/potion) model family. `go-potion` ports the original [model2vec](https://github.com/MinishLab/model2vec) library producing identical vectors, but encodes them an order of magnitude faster - mainly due to its native tokenizer library and use of simd. For single-threaded non-batch encoding it is over 8x faster than the original python and [rust](https://github.com/MinishLab/model2vec-rs) implementations.

## Usage

```bash
go get github.com/trengrj/go-potion
```

```go
import potion "github.com/trengrj/go-potion"

// downloads from Huggingface and caches the model on first use
encoder, err := potion.New(ctx, potion.BASE2M)

// encode a single sentence
embedding, err := encoder.Encode("hello world")

// encode a batch of sentences
embeddings, err := encoder.EncodeMany([]string{"first sentence", "second sentence"})
```

Embeddings are always l2-normalized (as every potion model is published with `normalize: true`, and go-potion matches model2vec's output), so the dot product of two embeddings is their cosine similarity — no normalization is needed before indexing or comparing them.

Model files are downloaded on first use and cached in the platform-native per-user cache directory (`~/Library/Caches/go-potion` on macOS, `~/.cache/go-potion` on Linux). To cache them somewhere else, set the `GO_POTION_HOME` environment variable.

## Model Variants

The library supports these models in the Minish Labs [potion collection](https://huggingface.co/collections/minishlab/potion):

| Constant | HuggingFace model | Dimensions | Size on disk | Notes |
|---|---|---|---|---|
| `BASE2M` | [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M) | 64 | 8 MB | Smallest model, fastest inference |
| `BASE4M` | [potion-base-4M](https://huggingface.co/minishlab/potion-base-4M) | 128 | 16 MB | Balanced size and performance |
| `BASE8M` | [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) | 256 | 31 MB | Most expressive base model |
| `BASE32M` | [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M) | 512 | 131 MB | Largest general-purpose model |
| `RETRIEVAL32M` | [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | 512 | 131 MB | Tuned for retrieval tasks |
| `SCIENCE32M` | [potion-science-32M](https://huggingface.co/minishlab/potion-science-32M) | 256 | 130 MB | Tuned for scientific text |
| `CODE16M` | [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) | 256 | 65 MB | Tuned for code; vocabulary-quantized (per-token weights + embedding row mapping) |
| `CODE16MV2` | [potion-code-16M-v2](https://huggingface.co/minishlab/potion-code-16M-v2) | 256 | 34 MB | Tuned for code; float16 embeddings (converted to float32 on load) |
| `MULTILINGUAL128M` | [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) | 256 | 531 MB | 101 languages; SentencePiece Unigram tokenizer instead of WordPiece |

Each model trades off between speed, memory usage, and embedding quality. All models share the WordPiece/BERT tokenization pipeline except `MULTILINGUAL128M`, which uses a SentencePiece Unigram pipeline (Precompiled charsmap normalization, Metaspace pre-tokenization, and Viterbi decoding) — go-potion implements both and picks the right one from the model's `tokenizer.json`.

## Development

Go tests compare tokenization and embeddings against reference output from the Python [model2vec](https://github.com/MinishLab/model2vec) library. The first run will download every supported model into the cache (~1.1 GB), so it takes a few minutes but subsequent runs are fast:

```bash
go test ./...
```

To regenerate new reference embeddings — after adding a model or bumping the model2vec version — run the following:

```bash
uv run validation/generate_tests.py  # regenerate validation/samples/*.json from model2vec
```

### Benchmarks

The `benchmarks/` folder holds the Python and Rust comparison benchmarks. All benchmarks process Peter Norvig's [big.txt](https://norvig.com/big.txt) (~6.2 MB of English text) in 256-word chunks, single-threaded:

```bash
go test -run '^$' -bench . -benchtime 5x                                        # Go: full encode + tokenize-only
uv run benchmarks/benchmark_encode_big_text.py 3                                # Python: model2vec full encode
cargo run --release --manifest-path benchmarks/rust-model2vec-bench/Cargo.toml  # Rust: model2vec-rs full encode
```

Each benchmark defaults to potion-base-2M and can run any other potion model — set `GO_POTION_BENCH_MODEL` for Go, or pass the HuggingFace model id to Python/Rust:

```bash
GO_POTION_BENCH_MODEL=RETRIEVAL32M go test -run '^$' -bench . -benchtime 5x
uv run benchmarks/benchmark_encode_big_text.py 3 minishlab/potion-retrieval-32M
cargo run --release --manifest-path benchmarks/rust-model2vec-bench/Cargo.toml -- 3 minishlab/potion-retrieval-32M
```

## Performance

Results on an Apple M1 Max (model2vec 0.8.2, model2vec-rs 0.2.1), with potion-base-2M (64 dimensions):

| Implementation | Work | Throughput |
|---|---|---|
| go-potion `Tokenize` | tokenize only | 62.6 MB/s (14.8M tokens/s) |
| go-potion `Encode` | tokenize + embed | 50.5 MB/s (35k vectors/s) |
| Rust [model2vec-rs](https://github.com/MinishLab/model2vec-rs) | tokenize + embed | 5.2 MB/s (3.7k vectors/s) |
| Python [model2vec](https://github.com/MinishLab/model2vec) | tokenize + embed | 3.9 MB/s (2.7k vectors/s) |

and with potion-retrieval-32M (512 dimensions):

| Implementation | Work | Throughput |
|---|---|---|
| go-potion `Tokenize` | tokenize only | 65.3 MB/s (14.8M tokens/s) |
| go-potion `Encode` | tokenize + embed | 29.1 MB/s (20.3k vectors/s) |
| Rust [model2vec-rs](https://github.com/MinishLab/model2vec-rs) | tokenize + embed | 3.7 MB/s (2.6k vectors/s) |
| Python [model2vec](https://github.com/MinishLab/model2vec) | tokenize + embed | 3.4 MB/s (2.4k vectors/s) |

go-potion and model2vec-rs do identical work — load the same `tokenizer.json`, emit essentially identical token IDs (~1.45M per pass over big.txt), filter `[UNK]`, mean-pool and L2-normalize — yet go-potion is ~10x faster on potion-base-2M. One main cause is model2vec-rs delegates tokenization to HuggingFace's [tokenizers](https://github.com/huggingface/tokenizers) crate (`encode_batch_fast`), and this library does alignment tracking which is not needed by static embeddings.

The static embedding of a sentence is just the l2-normalized mean of per-token vectors: token IDs go into a table lookup and are immediately reduced away. Offsets are never consulted, and there is no attention mask or padding because there is no sequence dimension, special tokens are not used (model2vec even filters `[UNK]` out), and token *order* does not affect the result. So `go-potion` implements only the forward text-to-IDs mapping as one-way passes: a fused single-pass normalizer with an ASCII fast path, zero-copy pre-tokenization into substrings, and allocation-free WordPiece lookups.

The mean-pooling loop is tuned as well: it accumulates in float32 (as model2vec and model2vec-rs both do for non-quantized models) directly into the output buffer, and picks its accumulation strategy by embedding width. Models with over 128 dimensions use SIMD ([github.com/tphakala/simd](https://github.com/tphakala/simd): NEON on arm64, SSE/AVX on amd64, pure-Go fallback elsewhere); narrower models, where per-row call overhead would outweigh vectorization, use a scalar loop that sums four rows per pass and is shaped so the compiler eliminates per-element bounds checks.

## License

MIT

