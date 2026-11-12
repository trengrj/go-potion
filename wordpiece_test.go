package potion

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

// newTestWordPiece builds a WordPiece tokenizer over a small synthetic
// vocabulary, bypassing tokenizer.json parsing.
func newTestWordPiece(vocab map[string]int, unkID, maxChars int) *wordPieceTokenizer {
	return newWordPiece(vocab, unkID, maxChars, *defaultBertNormalizer())
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

// TestTokenizeASCIIMatchesGeneral checks that the fused ASCII fast path
// yields exactly the tokens of the general normalize / pre-tokenize /
// WordPiece pipeline, over random ASCII text and every normalizer setting
// the fast path looks at.
func TestTokenizeASCIIMatchesGeneral(t *testing.T) {
	// Pieces around the pieceKeyBytes boundary, with continuation pieces,
	// upper-case entries (reachable only without lowercasing) and
	// punctuation both in and out of the vocabulary
	pieces := []string{
		"[UNK]", "a", "b", "c", "x", "0", "1", "ab", "abc", "the", "hello", "world",
		"abcdefg", "abcdefgh", "abcdefghi", "abcdefghijklmnop", "bcdefgha",
		"abcdefghabcdefgh", "abcdefghabcdefgha", "abcdefghabcdefghab",
		"##a", "##b", "##c", "##s", "##ab", "##ing", "##abcdef", "##abcdefgh", "##abcdefghi",
		"##abcdefghabcdefgh", "##abcdefghabcdefgha",
		"A", "AB", "Hello", "##S", "ABCDEFGHI",
		".", ",", "!", "'", "#", "[", "]", "-",
	}
	vocab := make(map[string]int, len(pieces))
	for i, piece := range pieces {
		vocab[piece] = i
	}

	// Mostly letters that form vocabulary words, plus every kind of byte
	// the fast path classifies: whitespace, punctuation, control
	// characters, DEL and NUL
	alphabet := []string{
		"a", "b", "c", "s", "x", "A", "B", "S", "0", "1", "ab", "abc", "the", "hello", "Hello",
		"abcdefgh", "abcdefghi", "ABCDEFGHI", "ing", "world",
		" ", " ", " ", "\t", "\n", "\r", "\v", "\f", ".", ",", "!", "'", "#", "?", "-",
		"\x00", "\x01", "\x1f", "\x7f",
	}

	rng := rand.New(rand.NewSource(1))
	for _, cleanText := range []bool{true, false} {
		for _, lowercase := range []bool{true, false} {
			for _, unkID := range []int{0, -1} {
				name := fmt.Sprintf("clean=%v/lower=%v/unk=%d", cleanText, lowercase, unkID)
				t.Run(name, func(t *testing.T) {
					normalizer := *newBertNormalizer(cleanText, true, lowercase, lowercase)
					tok := newWordPiece(vocab, unkID, 20, normalizer)
					fastHits := 0
					for i := 0; i < 20000; i++ {
						var sb strings.Builder
						for n := rng.Intn(12); n > 0; n-- {
							sb.WriteString(alphabet[rng.Intn(len(alphabet))])
						}
						input := sb.String()

						want, wantErr := tok.tokenizeGeneral(nil, input)
						got, ok, gotErr := tok.tokenizeASCII(nil, input)
						if !ok && gotErr == nil {
							continue // handed to the general pipeline
						}
						fastHits++
						if (gotErr != nil) != (wantErr != nil) {
							t.Fatalf("%q: fast path error %v, general error %v", input, gotErr, wantErr)
						}
						if gotErr != nil {
							if gotErr.Error() != wantErr.Error() {
								t.Fatalf("%q: fast path error %q, general error %q", input, gotErr, wantErr)
							}
							continue
						}
						if len(got) != len(want) || (len(got) > 0 && !reflect.DeepEqual(got, want)) {
							t.Fatalf("%q: fast path %v, general %v", input, got, want)
						}
					}
					if fastHits < 1000 {
						t.Fatalf("fast path handled only %d inputs", fastHits)
					}
				})
			}
		}
	}
}

// TestTokenizeASCIIFallsBack checks the inputs the fast path must hand to
// the general pipeline, and that the result is right end to end.
func TestTokenizeASCIIFallsBack(t *testing.T) {
	vocab := map[string]int{"[UNK]": 0, "ab": 1, "abcd": 2, "cd": 3, "caf": 4, "##e": 5, "cafe": 6}
	tok := newTestWordPiece(vocab, 0, 100)

	for _, tc := range []struct {
		input    string
		expected []int
	}{
		// A deleted control character joins its neighbours into one word
		{"ab\x01cd", []int{2}},
		{"ab cd\x00", []int{1, 3}},
		// Non-ASCII text needs accent stripping
		{"ab café", []int{1, 6}},
	} {
		if _, ok, _ := tok.tokenizeASCII(nil, tc.input); ok {
			t.Errorf("%q: fast path should have fallen back", tc.input)
		}
		got, err := tok.tokenize(tc.input)
		if err != nil {
			t.Fatalf("%q: %v", tc.input, err)
		}
		if !reflect.DeepEqual(got, tc.expected) {
			t.Errorf("%q: got %v, want %v", tc.input, got, tc.expected)
		}
	}
}

func TestPieceTable(t *testing.T) {
	// Lengths either side of the two key halves, keys that differ only
	// in trailing NUL bytes, and enough entries to force probe chains
	pieces := map[string]int{
		"a": 1, "a\x00": 2, "a\x00\x00": 3, "abcdefg": 4, "abcdefgh": 5, "abcdefgh\x00": 6,
		"abcdefghi": 7, "abcdefghijklmno": 8, "abcdefghijklmnop": 9, "héllo": 10,
	}
	for i := 0; i < 5000; i++ {
		pieces[fmt.Sprintf("w%d", i)] = 100 + i
	}
	table := newPieceTable(pieces)
	if len(table.tags) < 4*len(pieces) || len(table.tags) != len(table.entries) {
		t.Fatalf("table has %d tags and %d entries for %d pieces", len(table.tags), len(table.entries), len(pieces))
	}

	lookup := func(piece string) int {
		lo, hi := pieceKey(loadKey(piece, 0), loadKey(piece, 8), len(piece))
		return table.lookup(lo, hi, len(piece))
	}
	for piece, id := range pieces {
		if got := lookup(piece); got != id {
			t.Errorf("lookup(%q) = %d, want %d", piece, got, id)
		}
	}
	for _, absent := range []string{"b", "a\x00\x00\x00", "abcdefgi", "abcdefghijklmn", "abcdefghijklmnoq", "w5000", "W1"} {
		if got := lookup(absent); got != -1 {
			t.Errorf("lookup(%q) = %d, want -1", absent, got)
		}
	}
}

func TestScanWord(t *testing.T) {
	isAlnum := func(b byte) bool {
		return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
	}
	// Every byte value in every position, between neighbours that would
	// expose a carry between bytes
	for _, fill := range []byte{'a', 'Z', '0', '9', 'm'} {
		for b := 0; b < 0x100; b++ {
			for pos := 0; pos < 8; pos++ {
				var x uint64
				for k := 0; k < 8; k++ {
					c := fill
					if k == pos {
						c = byte(b)
					}
					x |= uint64(c) << (8 * k)
				}
				n, lowered := scanWord(x)

				wantN := 8
				if !isAlnum(byte(b)) {
					wantN = pos
				}
				if n != wantN {
					t.Fatalf("byte %#x at %d in %q fill: n = %d, want %d", b, pos, fill, n, wantN)
				}
				for k := 0; k < 8; k++ {
					c := byte(x >> (8 * k))
					want := c
					if c >= 'A' && c <= 'Z' {
						want += 'a' - 'A'
					}
					if got := byte(lowered >> (8 * k)); got != want {
						t.Fatalf("byte %#x at %d in %q fill: lowered[%d] = %#x, want %#x", b, pos, fill, k, got, want)
					}
				}
			}
		}
	}
}
