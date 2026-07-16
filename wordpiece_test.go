package potion

import (
	"encoding/json"
	"reflect"
	"testing"
)

// newTestWordPiece builds a WordPiece tokenizer over a small synthetic
// vocabulary, bypassing tokenizer.json parsing.
func newTestWordPiece(vocab map[string]int, unkID, maxChars int) *wordPieceTokenizer {
	return &wordPieceTokenizer{
		vocab:                vocab,
		singleByte:           singleByteIDs(vocab),
		unkID:                unkID,
		maxInputCharsPerWord: maxChars,
		normalizer:           *defaultBertNormalizer(),
	}
}

func TestPreTokenize(t *testing.T) {
	testCases := []struct {
		input    string
		expected []string
	}{
		{"", nil},
		{"hello", []string{"hello"}},
		{"hello world", []string{"hello", "world"}},
		{"  spaced   out  ", []string{"spaced", "out"}},
		{"hello!", []string{"hello", "!"}},
		// Every punctuation character is its own word, even in runs
		{"a...b", []string{"a", ".", ".", ".", "b"}},
		{"it's", []string{"it", "'", "s"}},
		// ASCII symbols in the BERT punctuation class
		{"1+2=3", []string{"1", "+", "2", "=", "3"}},
		{"$5", []string{"$", "5"}},
		// Unicode punctuation
		{"«quoted»", []string{"«", "quoted", "»"}},
		{"em—dash", []string{"em", "—", "dash"}},
		// Non-punctuation symbols stay attached
		{"a€b", []string{"a€b"}},
		{"...", []string{".", ".", "."}},
		// Non-ASCII whitespace splits words like ASCII space does
		{"a b", []string{"a", "b"}},
		{"a b", []string{"a", "b"}},
		{"tab\tsep", []string{"tab", "sep"}},
		// Multibyte runes inside and at the edges of words
		{"café tea", []string{"café", "tea"}},
		{"—start", []string{"—", "start"}},
		{"end—", []string{"end", "—"}},
		{"日本語", []string{"日本語"}},
		{"mixé!", []string{"mixé", "!"}},
	}

	for _, tc := range testCases {
		got := preTokenize(tc.input)
		if len(got) == 0 && len(tc.expected) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, tc.expected) {
			t.Errorf("preTokenize(%q) = %v, want %v", tc.input, got, tc.expected)
		}
	}
}

func TestWordPieceTokenize(t *testing.T) {
	vocab := map[string]int{
		"[UNK]":  0,
		"hello":  1,
		"world":  2,
		"!":      3,
		"un":     4,
		"##aff":  5,
		"##able": 6,
		"'":      7,
		"s":      8,
		"it":     9,
	}

	testCases := []struct {
		name     string
		input    string
		expected []int
	}{
		{"empty", "", []int{}},
		{"single word", "hello", []int{1}},
		{"normalized casing", "HELLO World!", []int{1, 2, 3}},
		{"greedy subword continuation", "unaffable", []int{4, 5, 6}},
		{"punctuation split", "it's", []int{9, 7, 8}},
		// A word with no full tokenization maps to [UNK], which is
		// dropped like model2vec does
		{"unknown word dropped", "hello xyzzy world", []int{1, 2}},
		{"unknown single char dropped", "hello & world", []int{1, 2}},
		{"unknown single letter dropped", "hello z world", []int{1, 2}},
		// Partial subword matches must not leak when the tail fails
		{"partial match dropped whole", "hello unaffordable world", []int{1, 2}},
		{"only unknown", "xyzzy", []int{}},
	}

	tok := newTestWordPiece(vocab, 0, 100)
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.tokenize(tc.input)
			if err != nil {
				t.Fatalf("tokenize(%q): %v", tc.input, err)
			}
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("tokenize(%q) = %v, want %v", tc.input, got, tc.expected)
			}
		})
	}
}

func TestWordPieceMaxInputChars(t *testing.T) {
	vocab := map[string]int{"[UNK]": 0, "un": 1, "##aff": 2, "##able": 3}

	// "unaffable" is 9 characters; with the limit below it, the whole
	// word maps to [UNK] and is dropped
	tok := newTestWordPiece(vocab, 0, 5)
	got, err := tok.tokenize("unaffable")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected over-long word to be dropped, got %v", got)
	}

	tok = newTestWordPiece(vocab, 0, 9)
	got, err = tok.tokenize("unaffable")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if want := []int{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("tokenize = %v, want %v", got, want)
	}

	// The limit counts runes, not bytes: "ααααα" is 10 bytes but 5 runes,
	// so it must survive a limit of 5
	tok = newTestWordPiece(map[string]int{"[UNK]": 0, "αα": 1, "##ααα": 2}, 0, 5)
	got, err = tok.tokenize("ααααα")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if want := []int{1, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("tokenize = %v, want %v", got, want)
	}
}

func TestWordPieceNoUnknownToken(t *testing.T) {
	tok := newTestWordPiece(map[string]int{"hello": 1}, -1, 100)

	if _, err := tok.tokenize("hello"); err != nil {
		t.Errorf("known word should tokenize without unk token: %v", err)
	}
	if _, err := tok.tokenize("xyzzy"); err == nil {
		t.Error("expected error tokenizing unknown word without unk token")
	}
}

func TestNewWordPieceTokenizer(t *testing.T) {
	file := tokenizerFile{
		Normalizer: json.RawMessage(`{
			"type": "BertNormalizer",
			"clean_text": true,
			"handle_chinese_chars": true,
			"strip_accents": null,
			"lowercase": true
		}`),
		Model: json.RawMessage(`{
			"type": "WordPiece",
			"unk_token": "[UNK]",
			"max_input_chars_per_word": 50,
			"vocab": {"[UNK]": 7, "hello": 1}
		}`),
	}

	tok, err := newWordPieceTokenizer(file)
	if err != nil {
		t.Fatalf("newWordPieceTokenizer: %v", err)
	}
	if tok.unkID != 7 {
		t.Errorf("unkID = %d, want 7", tok.unkID)
	}
	if tok.maxInputCharsPerWord != 50 {
		t.Errorf("maxInputCharsPerWord = %d, want 50", tok.maxInputCharsPerWord)
	}
	// strip_accents: null follows the lowercase setting
	if !tok.normalizer.StripAccents {
		t.Error("expected StripAccents to default to the lowercase setting")
	}

	// Missing optional fields fall back to defaults
	file.Model = json.RawMessage(`{"type": "WordPiece", "vocab": {"hi": 0}}`)
	file.Normalizer = nil
	tok, err = newWordPieceTokenizer(file)
	if err != nil {
		t.Fatalf("newWordPieceTokenizer with minimal config: %v", err)
	}
	if tok.unkID != -1 {
		t.Errorf("unkID = %d, want -1 when no unk token is configured", tok.unkID)
	}
	if tok.maxInputCharsPerWord != 100 {
		t.Errorf("maxInputCharsPerWord = %d, want default 100", tok.maxInputCharsPerWord)
	}
}
