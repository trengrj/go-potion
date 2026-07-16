package potion

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// metaspaceJSON is the pre-tokenizer configuration potion-multilingual-128M
// ships, reused across tests.
const metaspaceJSON = `{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false}`

// newTestUnigram builds a Unigram tokenizer from raw tokenizer.json
// sections, so tests exercise the same construction path as real models.
func newTestUnigram(t *testing.T, modelJSON, preJSON, normalizerJSON string) *unigramTokenizer {
	t.Helper()
	file := tokenizerFile{
		Model:        json.RawMessage(modelJSON),
		PreTokenizer: json.RawMessage(preJSON),
	}
	if normalizerJSON != "" {
		file.Normalizer = json.RawMessage(normalizerJSON)
	}
	tok, err := newUnigramTokenizer(file)
	if err != nil {
		t.Fatalf("newUnigramTokenizer: %v", err)
	}
	return tok
}

func TestUnigramViterbi(t *testing.T) {
	// The doc-test vocabulary of HuggingFace's Unigram model
	// (models/unigram/model.rs): encode("abcdacdxx") must yield
	// ["abcd", "a", "cd", "xx"], the "xx" being two fused unknowns
	tok := newTestUnigram(t, `{
		"type": "Unigram",
		"unk_id": 0,
		"vocab": [
			["<unk>", 0], ["a", 0], ["b", 0], ["c", 0], ["d", 0],
			["cd", 1], ["ab", 2], ["abc", 5], ["abcd", 10]
		],
		"byte_fallback": false
	}`, metaspaceJSON, "")

	pieces, err := tok.viterbi("abcdacdxx")
	if err != nil {
		t.Fatalf("viterbi: %v", err)
	}
	if want := []string{"abcd", "a", "cd", "xx"}; !reflect.DeepEqual(pieces, want) {
		t.Errorf("viterbi(abcdacdxx) = %v, want %v", pieces, want)
	}

	// The fused unknown run "xx" has no vocabulary entry and no byte
	// fallback, so it maps to the unknown ID
	ids, err := tok.tokenize("abcdacdxx")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	// Metaspace prepends "▁", which is unknown too and fuses with
	// nothing adjacent (abcd follows immediately)
	if want := []int{0, 8, 1, 5, 0}; !reflect.DeepEqual(ids, want) {
		t.Errorf("tokenize(abcdacdxx) = %v, want %v", ids, want)
	}
}

func TestUnigramViterbiTieBreak(t *testing.T) {
	// "a"+"b" and "ab" both score -1.0; HuggingFace keeps the node
	// written first (the single token spanning from the earlier start),
	// so equal-score ties must resolve to "ab"
	tok := newTestUnigram(t, `{
		"type": "Unigram",
		"unk_id": 0,
		"vocab": [["<unk>", 0], ["a", -0.5], ["b", -0.5], ["ab", -1.0]],
		"byte_fallback": false
	}`, metaspaceJSON, "")

	pieces, err := tok.viterbi("ab")
	if err != nil {
		t.Fatalf("viterbi: %v", err)
	}
	if want := []string{"ab"}; !reflect.DeepEqual(pieces, want) {
		t.Errorf("viterbi(ab) = %v, want %v", pieces, want)
	}
}

func TestUnigramTokenize(t *testing.T) {
	tok := newTestUnigram(t, `{
		"type": "Unigram",
		"unk_id": 0,
		"vocab": [
			["<unk>", 0], ["▁", -1], ["▁hello", -1], ["▁world", -1],
			["!", -1], ["▁h", -2], ["ello", -2]
		],
		"byte_fallback": false
	}`, metaspaceJSON, "")

	testCases := []struct {
		name     string
		input    string
		expected []int
	}{
		{"prepended meta char", "hello", []int{2}},
		{"spaces become meta chars", "hello world!", []int{2, 3, 4}},
		// "▁hello" (-1) must beat "▁h"+"ello" (-4)
		{"best segmentation wins", "hello!", []int{2, 4}},
		{"unknown char", "hello ~", []int{2, 1, 0}},
		// Consecutive unknowns fuse into a single unk token
		{"fused unknowns", "hello ~~", []int{2, 1, 0}},
		{"whitespace only", "   ", []int{1, 1, 1}},
	}

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

func TestUnigramByteFallback(t *testing.T) {
	tok := newTestUnigram(t, `{
		"type": "Unigram",
		"unk_id": 0,
		"vocab": [["<unk>", 0], ["▁", -1], ["<0x78>", -5], ["a", -1]],
		"byte_fallback": true
	}`, metaspaceJSON, "")

	// "x" has no piece but its byte token <0x78> exists
	ids, err := tok.tokenize("a x")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if want := []int{1, 3, 1, 2}; !reflect.DeepEqual(ids, want) {
		t.Errorf("tokenize(a x) = %v, want %v", ids, want)
	}

	// "é" (0xC3 0xA9) has no byte tokens, so it falls back to unk
	ids, err = tok.tokenize("é")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if want := []int{1, 0}; !reflect.DeepEqual(ids, want) {
		t.Errorf("tokenize(é) = %v, want %v", ids, want)
	}
}

func TestUnigramNormalizerChain(t *testing.T) {
	// A miniature of the multilingual model's chain: pad punctuation
	// with spaces, collapse whitespace, strip
	tok := newTestUnigram(t, `{
		"type": "Unigram",
		"unk_id": 0,
		"vocab": [["<unk>", 0], ["▁", -1], ["▁hi", -1], ["!", -1]],
		"byte_fallback": false
	}`, metaspaceJSON, `{
		"type": "Sequence",
		"normalizers": [
			{"type": "Replace", "pattern": {"String": "!"}, "content": " ! "},
			{"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
			{"type": "Strip", "strip_left": true, "strip_right": true}
		]
	}`)

	// "  hi!! " -> "hi ! !" -> "▁hi▁!▁!"
	ids, err := tok.tokenize("  hi!! ")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if want := []int{2, 1, 3, 1, 3}; !reflect.DeepEqual(ids, want) {
		t.Errorf("tokenize = %v, want %v", ids, want)
	}

	// A fully stripped input produces no tokens
	ids, err = tok.tokenize("   ")
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if len(ids) != 0 {
		t.Errorf("tokenize(whitespace) = %v, want none", ids)
	}
}

func TestReplaceNormalizerUnicodeWhitespace(t *testing.T) {
	// Rust's regex \s matches Unicode White_Space; the Go translation
	// must cover characters outside Go's ASCII-only \s
	steps, err := parseSPNormalizers(json.RawMessage(`{
		"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "
	}`))
	if err != nil {
		t.Fatalf("parseSPNormalizers: %v", err)
	}
	if len(steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(steps))
	}

	// Ideographic space, no-break space, line separator, ASCII run
	got := steps[0].normalize("a\u3000\u00A0b c\u2028 \t d")
	if want := "a b c d"; got != want {
		t.Errorf("normalize = %q, want %q", got, want)
	}
}

func TestParseSPNormalizers(t *testing.T) {
	// Nested sequences flatten in order
	steps, err := parseSPNormalizers(json.RawMessage(`{
		"type": "Sequence",
		"normalizers": [
			{"type": "Sequence", "normalizers": [
				{"type": "Replace", "pattern": {"String": "a"}, "content": "b"}
			]},
			{"type": "Replace", "pattern": {"String": "bb"}, "content": "c"}
		]
	}`))
	if err != nil {
		t.Fatalf("parseSPNormalizers: %v", err)
	}
	if len(steps) != 2 {
		t.Fatalf("expected 2 flattened steps, got %d", len(steps))
	}
	s := "ab"
	for _, step := range steps {
		s = step.normalize(s)
	}
	if s != "c" {
		t.Errorf("chained normalize = %q, want %q", s, "c")
	}

	// Unknown types must fail loudly
	if _, err := parseSPNormalizers(json.RawMessage(`{"type": "NFKC"}`)); err == nil {
		t.Error("expected error for unsupported normalizer type")
	}

	// Absent section is fine
	steps, err = parseSPNormalizers(nil)
	if err != nil || steps != nil {
		t.Errorf("parseSPNormalizers(nil) = %v, %v; want nil, nil", steps, err)
	}
}

func TestStripNormalizer(t *testing.T) {
	n := &stripNormalizer{left: true, right: false}
	if got := n.normalize("  a  "); got != "a  " {
		t.Errorf("left strip = %q", got)
	}
	n = &stripNormalizer{left: false, right: true}
	if got := n.normalize("  a　"); got != "  a" {
		t.Errorf("right strip = %q", got)
	}
}

func TestSentencePieceUnmarshal(t *testing.T) {
	var p sentencePiece
	if err := json.Unmarshal([]byte(`["tok", -1.5]`), &p); err != nil {
		t.Fatalf("valid entry: %v", err)
	}
	if p.token != "tok" || p.score != -1.5 {
		t.Errorf("got %+v", p)
	}

	for _, bad := range []string{`[1, -1.5]`, `["tok", "x"]`, `["tok"]`, `"tok"`} {
		if err := json.Unmarshal([]byte(bad), &p); err == nil {
			t.Errorf("expected error for %s", bad)
		}
	}
}

func TestUnigramConfigErrors(t *testing.T) {
	model := `{"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0]], "byte_fallback": false}`

	cases := []struct {
		name    string
		model   string
		pre     string
		wantErr string
	}{
		{
			name:    "splitting metaspace",
			model:   model,
			pre:     `{"type": "Metaspace", "replacement": "▁", "split": true}`,
			wantErr: "split=true",
		},
		{
			name:    "non-metaspace pre-tokenizer",
			model:   model,
			pre:     `{"type": "Whitespace"}`,
			wantErr: "unsupported pre-tokenizer",
		},
		{
			name:    "unk_id outside vocabulary",
			model:   `{"type": "Unigram", "unk_id": 5, "vocab": [["<unk>", 0]], "byte_fallback": false}`,
			pre:     metaspaceJSON,
			wantErr: "outside vocabulary",
		},
		{
			name:    "empty vocabulary",
			model:   `{"type": "Unigram", "unk_id": 0, "vocab": [], "byte_fallback": false}`,
			pre:     metaspaceJSON,
			wantErr: "empty vocabulary",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := newUnigramTokenizer(tokenizerFile{
				Model:        json.RawMessage(tc.model),
				PreTokenizer: json.RawMessage(tc.pre),
			})
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("expected error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestUnigramNoUnknownToken(t *testing.T) {
	// Without an unk_id, a character with no vocabulary entry is an error
	tok := newTestUnigram(t, `{
		"type": "Unigram",
		"vocab": [["▁", -1], ["a", -1]],
		"byte_fallback": false
	}`, metaspaceJSON, "")

	if _, err := tok.tokenize("a"); err != nil {
		t.Errorf("known input should tokenize: %v", err)
	}
	if _, err := tok.tokenize("z"); err == nil {
		t.Error("expected error for unknown character without unk token")
	}
}
