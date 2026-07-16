use std::env;
use std::fs;
use std::time::Instant;

use model2vec_rs::model::StaticModel;

// Mirrors go-potion's BenchmarkEncodeBigText and the Python benchmark:
// full encode (tokenize + embed) of big.txt in pre-built 256-word chunks,
// one chunk per encode call, single-threaded, with potion-base-2M by
// default (pass another model id as the second argument).
//
// Reads big.txt from go-potion's shared cache, so run the Go benchmark
// once first to populate it. The model itself is loaded through
// model2vec-rs's own HuggingFace hub support (cached by hf-hub):
//
//   go test -run '^$' -bench BenchmarkEncodeBigText -benchtime 1x
//   cargo run --release --manifest-path benchmarks/rust-model2vec-bench/Cargo.toml -- 3
//   cargo run --release --manifest-path benchmarks/rust-model2vec-bench/Cargo.toml -- 3 minishlab/potion-retrieval-32M
fn main() {
    let iters: usize = env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(3);
    let model_id = env::args()
        .nth(2)
        .unwrap_or_else(|| "minishlab/potion-base-2M".to_string());

    let cache = cache_dir();
    let text = fs::read_to_string(format!("{cache}/big.txt"))
        .expect("big.txt not found in cache; run the Go benchmark first");
    let words: Vec<&str> = text.split_whitespace().collect();
    let chunks: Vec<String> = words.chunks(256).map(|c| c.join(" ")).collect();
    let total_bytes: usize = chunks.iter().map(|c| c.len()).sum();

    let model = StaticModel::from_pretrained(&model_id, None, None, None)
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));

    let mut total_vectors = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        for chunk in &chunks {
            let embeddings = model.encode(std::slice::from_ref(chunk));
            total_vectors += embeddings.len();
        }
    }
    report("encode", start.elapsed().as_secs_f64(), total_bytes * iters, total_vectors);
}

// cache_dir mirrors go-potion's resolveCacheDir: GO_POTION_HOME, then the
// platform user cache directory.
fn cache_dir() -> String {
    if let Ok(dir) = env::var("GO_POTION_HOME") {
        if !dir.is_empty() {
            return dir;
        }
    }
    if cfg!(target_os = "macos") {
        let home = env::var("HOME").expect("HOME not set");
        return format!("{home}/Library/Caches/go-potion");
    }
    if cfg!(target_os = "windows") {
        let local = env::var("LOCALAPPDATA").expect("LOCALAPPDATA not set");
        return format!("{local}\\go-potion");
    }
    if let Ok(xdg) = env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return format!("{xdg}/go-potion");
        }
    }
    let home = env::var("HOME").expect("HOME not set");
    format!("{home}/.cache/go-potion")
}

fn report(name: &str, secs: f64, bytes: usize, vectors: usize) {
    println!(
        "{name}: {:.2} MB/s, {:.0} vectors/s ({vectors} vectors, {:.3}s)",
        bytes as f64 / secs / 1024.0 / 1024.0,
        vectors as f64 / secs,
        secs
    );
}
