package potion

import (
	"encoding/json"
	"fmt"
	"unicode"
	"unicode/utf8"
)

// wordPieceConfig is the tokenizer.json layout shared by the BERT-style
// POTION models: a BertNormalizer section and a WordPiece model section.
type wordPieceConfig struct {
	Normalizer *struct {
		Type               string `json:"type"`
		CleanText          bool   `json:"clean_text"`
		HandleChineseChars bool   `json:"handle_chinese_chars"`
		// null means "follow the lowercase setting", matching HuggingFace
		StripAccents *bool `json:"strip_accents"`
		Lowercase    bool  `json:"lowercase"`
	}
	Model struct {
		Vocab                map[string]int `json:"vocab"`
		UnkToken             string         `json:"unk_token,omitempty"`
		MaxInputCharsPerWord int            `json:"max_input_chars_per_word,omitempty"`
	}
}

// wordPieceTokenizer implements the HuggingFace BERT pipeline used by every
// POTION model except potion-multilingual-128M: BertNormalizer,
// BertPreTokenizer, then greedy WordPiece. Words that map to the unknown
// token are dropped, matching model2vec, which filters unknown tokens out
// before averaging embeddings.
type wordPieceTokenizer struct {
	vocab map[string]int
	// singleByte caches the vocab IDs of single-ASCII-byte tokens (-1 when
	// absent), so the isolated punctuation words the pre-tokenizer produces
	// skip the map lookup.
	singleByte           [128]int32
	unkID                int // -1 if the vocabulary has no unknown token
	maxInputCharsPerWord int
	normalizer           bertNormalizer
}

// singleByteIDs builds the singleByte cache from a vocabulary.
func singleByteIDs(vocab map[string]int) (t [128]int32) {
	for b := range t {
		t[b] = -1
		if id, ok := vocab[string(rune(b))]; ok {
			t[b] = int32(id)
		}
	}
	return t
}

// newWordPieceTokenizer builds a wordPieceTokenizer from a parsed
// tokenizer.json file.
func newWordPieceTokenizer(file tokenizerFile) (*wordPieceTokenizer, error) {
	var cfg wordPieceConfig
	if len(file.Normalizer) > 0 {
		if err := json.Unmarshal(file.Normalizer, &cfg.Normalizer); err != nil {
			return nil, fmt.Errorf("normalizer section: %w", err)
		}
	}
	if err := json.Unmarshal(file.Model, &cfg.Model); err != nil {
		return nil, fmt.Errorf("model section: %w", err)
	}

	unkID := -1
	if cfg.Model.UnkToken != "" {
		if id, ok := cfg.Model.Vocab[cfg.Model.UnkToken]; ok {
			unkID = id
		}
	}

	maxInputCharsPerWord := cfg.Model.MaxInputCharsPerWord
	if maxInputCharsPerWord <= 0 {
		maxInputCharsPerWord = 100
	}

	return &wordPieceTokenizer{
		vocab:                cfg.Model.Vocab,
		singleByte:           singleByteIDs(cfg.Model.Vocab),
		unkID:                unkID,
		maxInputCharsPerWord: maxInputCharsPerWord,
		normalizer:           *bertNormalizerFromConfig(cfg),
	}, nil
}

// bertNormalizerFromConfig builds a bertNormalizer from the tokenizer.json
// normalizer section, falling back to the defaults when absent
func bertNormalizerFromConfig(cfg wordPieceConfig) *bertNormalizer {
	n := cfg.Normalizer
	if n == nil || n.Type != "BertNormalizer" {
		return defaultBertNormalizer()
	}
	stripAccents := n.Lowercase
	if n.StripAccents != nil {
		stripAccents = *n.StripAccents
	}
	return newBertNormalizer(n.CleanText, n.HandleChineseChars, stripAccents, n.Lowercase)
}

// isBertPunctuation matches HuggingFace's BertPreTokenizer punctuation class:
// all ASCII punctuation (which includes symbols like $, +, <, =, >) plus every
// Unicode punctuation category.
func isBertPunctuation(r rune) bool {
	if (r >= '!' && r <= '/') || (r >= ':' && r <= '@') || (r >= '[' && r <= '`') || (r >= '{' && r <= '~') {
		return true
	}
	return unicode.IsPunct(r)
}

// ASCII byte classes for preTokenize's fast path.
const (
	asciiWord byte = iota
	asciiSpace
	asciiPunct
)

// asciiClass classifies each ASCII byte the way the rune-level predicates
// below classify it: unicode.IsSpace and isBertPunctuation agree with this
// table on every value under 0x80.
var asciiClass = func() (t [128]byte) {
	for b := range t {
		switch {
		case isASCIISpace(byte(b)):
			t[b] = asciiSpace
		case (b >= '!' && b <= '/') || (b >= ':' && b <= '@') || (b >= '[' && b <= '`') || (b >= '{' && b <= '~'):
			t[b] = asciiPunct
		}
	}
	return t
}()

// preTokenize splits normalized text like HuggingFace's BertPreTokenizer:
// split on whitespace (removed), then isolate each punctuation character as
// its own word. The returned words are zero-copy substrings of sentence.
// ASCII bytes — the entire input for most POTION models, whose normalizer
// lowercases to ASCII-heavy text — classify through a table without rune
// decoding; only bytes >= 0x80 take the Unicode path.
func preTokenize(sentence string) []string {
	words := make([]string, 0, len(sentence)/4)
	start := -1

	for i := 0; i < len(sentence); {
		var class byte
		size := 1
		if b := sentence[i]; b < utf8.RuneSelf {
			class = asciiClass[b]
		} else {
			var r rune
			r, size = utf8.DecodeRuneInString(sentence[i:])
			switch {
			case unicode.IsSpace(r):
				class = asciiSpace
			case isBertPunctuation(r):
				class = asciiPunct
			}
		}

		switch class {
		case asciiSpace:
			if start >= 0 {
				words = append(words, sentence[start:i])
				start = -1
			}
		case asciiPunct:
			if start >= 0 {
				words = append(words, sentence[start:i])
				start = -1
			}
			words = append(words, sentence[i:i+size])
		default:
			if start < 0 {
				start = i
			}
		}
		i += size
	}
	if start >= 0 {
		words = append(words, sentence[start:])
	}

	return words
}

// word2tok appends the WordPiece token IDs of a single word to dst using
// greedy longest-match-first, and returns the extended slice. A word that
// cannot be fully tokenized (or exceeds maxInputCharsPerWord) maps to the
// unknown token as a whole in the reference implementation; model2vec then
// drops unknown tokens since their embedding carries no signal, so such
// words contribute nothing to dst.
func (t *wordPieceTokenizer) word2tok(dst []int, word string) ([]int, error) {
	mark := len(dst)

	// Rune count never exceeds byte length, so words short enough in bytes
	// skip the counting scan entirely
	if len(word) > t.maxInputCharsPerWord && utf8.RuneCountInString(word) > t.maxInputCharsPerWord {
		return t.dropUnknown(dst, mark, word)
	}

	if len(word) == 1 && word[0] < utf8.RuneSelf {
		if id := t.singleByte[word[0]]; id >= 0 {
			return append(dst, int(id)), nil
		}
		return t.dropUnknown(dst, mark, word)
	}

	// Scratch buffer for "##"-prefixed lookups; the vocab access via
	// string(buf) does not allocate
	var scratch [512]byte

	start := 0
	for start < len(word) {
		end := len(word)
		token := -1
		for end > start {
			var id int
			var ok bool
			if start == 0 {
				id, ok = t.vocab[word[:end]]
			} else {
				buf := append(append(scratch[:0], '#', '#'), word[start:end]...)
				id, ok = t.vocab[string(buf)]
			}
			if ok {
				token = id
				break
			}
			_, size := utf8.DecodeLastRuneInString(word[start:end])
			end -= size
		}
		if token == -1 {
			return t.dropUnknown(dst, mark, word)
		}
		dst = append(dst, token)
		start = end
	}

	return dst, nil
}

// dropUnknown discards any partial tokens appended for word. This matches
// mapping the whole word to the unknown token and filtering it out.
func (t *wordPieceTokenizer) dropUnknown(dst []int, mark int, word string) ([]int, error) {
	if t.unkID >= 0 {
		return dst[:mark], nil
	}
	return nil, fmt.Errorf("cannot tokenize word and no unknown token available: %s", word)
}

func (t *wordPieceTokenizer) tokenize(sentence string) ([]int, error) {
	normalized := t.normalizer.normalize(sentence)
	words := preTokenize(normalized)
	allTokens := make([]int, 0, len(words))

	var err error
	for _, word := range words {
		allTokens, err = t.word2tok(allTokens, word)
		if err != nil {
			return nil, err
		}
	}

	return allTokens, nil
}
