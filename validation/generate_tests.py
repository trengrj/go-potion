# /// script
# requires-python = ">=3.10"
# dependencies = ["model2vec"]
# ///
"""Generate reference tokenization and embedding samples with model2vec.

Writes one samples/<MODEL>.json per model into this directory. The Go test
suite (potion_test.go) compares its own output against these files.

Run from the repository root with: uv run validation/generate_tests.py
"""

import json
from pathlib import Path

from model2vec import StaticModel

# Every model in the POTION collection
# (https://huggingface.co/collections/minishlab/potion)
MODELS = {
    "BASE2M": "minishlab/potion-base-2M",
    "BASE4M": "minishlab/potion-base-4M",
    "BASE8M": "minishlab/potion-base-8M",
    "BASE32M": "minishlab/potion-base-32M",
    "RETRIEVAL32M": "minishlab/potion-retrieval-32M",
    "SCIENCE32M": "minishlab/potion-science-32M",
    "CODE16M": "minishlab/potion-code-16M",
    "CODE16MV2": "minishlab/potion-code-16M-v2",
    "MULTILINGUAL128M": "minishlab/potion-multilingual-128M",
}

# potion-code-16M-v2 ships float16 embeddings; numpy would then average in
# half precision, which the float32/float64 Go implementation can't (and
# shouldn't) reproduce bit-for-bit. Upcasting on load keeps the reference
# computation in float32 like every other model.
QUANTIZE_TO = {
    "CODE16MV2": "float32",
}

EXAMPLES = [
    # English
    "test",
    "hello world",
    "hello world!",
    "Hello world!",
    "Unnnknown character",
    "It's dangerous to go alone",
    "Someone asks you to “make some embeddings”. What do you input? You input text.1 You don’t need to provide the same amount of text every time. E.g. sometimes your input is a single paragraph while at other times it’s a few sections, an entire document, or even multiple documents.",
    # Code (the potion-code models target this domain)
    "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
    'if err != nil {\n\treturn fmt.Errorf("failed to open %s: %w", path, err)\n}',
    # German (umlauts, eszett)
    "Es ist gefährlich, alleine zu gehen",
    "Die Straße zu überqueren ist größer als gedacht, viele Grüße",
    # French (accents, apostrophes)
    "L'été dernier, j'ai visité un château près de la côte",
    # Spanish (inverted punctuation, ñ)
    "¿Dónde está la biblioteca? ¡El niño llegará mañana!",
    # Portuguese (tilde, cedilla)
    "A canção do coração não tem tradução",
    # Turkish (dotted capital İ, ğ, ç)
    "İstanbul Boğazı'nda balık tutmak çok güzel",
    # Vietnamese (stacked diacritics)
    "Tiếng Việt rất thú vị",
    # Russian (Cyrillic)
    "Москва — столица России",
    # Greek
    "Η γρήγορη αλεπού πηδάει πάνω από τον σκύλο",
    # Japanese (kanji + hiragana + katakana)
    "一人で行くのは危険です",
    "東京タワーは高いです",
    # Chinese
    "我爱北京天安门",
    # Korean (Hangul; NFD decomposes syllables into jamo)
    "한국어를 배우고 있어요",
    # Arabic (right-to-left)
    "اللغة العربية جميلة",
    # Hebrew (right-to-left)
    "שלום עולם",
    # Hindi (Devanagari)
    "नमस्ते दुनिया",
    # Thai (no word boundaries)
    "สวัสดีชาวโลก",
    # Emoji and mixed scripts
    "I ❤️ embeddings 🚀 vectors!",
    "Mixed languages: hello, 世界, мир, and عالم together",
]


def main() -> None:
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)

    for name, repo in MODELS.items():
        print(f"Generating samples for {name} ({repo})...")
        model = StaticModel.from_pretrained(repo, quantize_to=QUANTIZE_TO.get(name))

        output = []
        for example in EXAMPLES:
            output.append(
                {
                    "text": example,
                    "embedding": model.encode(example).tolist(),
                    "tokens": model.tokenize([example])[0],
                }
            )

        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
