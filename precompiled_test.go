package potion

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// newTestPrecompiled hand-builds a minimal charsmap blob with three
// entries, exercising the Darts double-array encoding directly:
//
//	"A" (0x41)            -> "x"
//	"B" (0x42)            -> "yz"
//	U+0301 (0xCC 0x81)    -> "" (deletion)
//
// The double-array units are placed manually: the root has offset 256, so
// a first byte c lands at position 256^c; leaf units carry bit 8 and an
// offset to their value unit, which stores the byte offset of the
// replacement inside the NUL-separated pool "x\0yz\0\0".
func newTestPrecompiled(t *testing.T) *precompiled {
	t.Helper()

	trie := make([]uint32, 512)
	trie[0] = 256 << 10              // root: offset 256
	trie[321] = 1<<10 | 0x100 | 0x41 // 'A': leaf, value unit at 321^1
	trie[320] = 1 << 31              // value: pool offset 0 ("x")
	trie[322] = 6<<10 | 0x100 | 0x42 // 'B': leaf, value unit at 322^6
	trie[324] = 1<<31 | 2            // value: pool offset 2 ("yz")
	trie[460] = 16<<10 | 0xCC        // 0xCC: interior, next level at 460^16
	trie[349] = 2<<10 | 0x100 | 0x81 // 0x81: leaf, value unit at 349^2
	trie[351] = 1<<31 | 5            // value: pool offset 5 ("")

	blob := make([]byte, 4+len(trie)*4)
	binary.LittleEndian.PutUint32(blob, uint32(len(trie)*4))
	for i, u := range trie {
		binary.LittleEndian.PutUint32(blob[4+i*4:], u)
	}
	blob = append(blob, "x\x00yz\x00\x00"...)

	p, err := newPrecompiled(base64.StdEncoding.EncodeToString(blob))
	if err != nil {
		t.Fatalf("newPrecompiled: %v", err)
	}
	return p
}

func TestPrecompiledTransform(t *testing.T) {
	p := newTestPrecompiled(t)

	testCases := []struct {
		chunk string
		want  string
		ok    bool
	}{
		{"A", "x", true},
		{"B", "yz", true},
		{"\u0301", "", true}, // deletion entry
		{"C", "", false},     // no entry
		{"", "", false},
		// The transform takes the FIRST (shortest) prefix match,
		// like spm_precompiled's common_prefix_search
		{"AB", "x", true},
		// A NUL byte stops the trie walk
		{"\x00A", "", false},
	}

	for _, tc := range testCases {
		got, ok := p.transform(tc.chunk)
		if ok != tc.ok || got != tc.want {
			t.Errorf("transform(%q) = %q, %v; want %q, %v", tc.chunk, got, ok, tc.want, tc.ok)
		}
	}
}

func TestPrecompiledNormalize(t *testing.T) {
	p := newTestPrecompiled(t)

	testCases := []struct {
		input string
		want  string
	}{
		{"", ""},
		{"AB", "xyz"},
		{"eA", "ex"},
		// "e"+U+0301 is one grapheme (3 bytes, under the whole-grapheme
		// threshold) with no entry as a whole, so each character is
		// transformed independently and the combining mark is deleted
		{"e\u0301A", "ex"},
		// "A"+U+0301 as a whole grapheme prefix-matches the "A" entry
		{"A\u0301", "x"},
		{"nope", "nope"},
	}

	for _, tc := range testCases {
		if got := p.normalize(tc.input); got != tc.want {
			t.Errorf("normalize(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// realMultilingualNormalizer downloads (or reuses from the shared model
// cache) the potion-multilingual-128M tokenizer.json and returns its parsed
// normalizer chain.
func realMultilingualNormalizer(t *testing.T) []spNormalizer {
	t.Helper()
	cacheDir := testCacheDir(t)

	modelDir := filepath.Join(cacheDir, string(MULTILINGUAL128M))
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	tokenizerPath := filepath.Join(modelDir, "tokenizer.json")
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/tokenizer.json", modelRepos[MULTILINGUAL128M])
	if err := ensureFile(t.Context(), url, tokenizerPath); err != nil {
		t.Fatalf("ensureFile: %v", err)
	}

	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		t.Fatalf("read tokenizer.json: %v", err)
	}
	var file tokenizerFile
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("parse tokenizer.json: %v", err)
	}
	steps, err := parseSPNormalizers(file.Normalizer)
	if err != nil {
		t.Fatalf("parseSPNormalizers: %v", err)
	}
	return steps
}

// TestPrecompiledRealCharsmap runs potion-multilingual-128M's actual
// compiled charsmap against inputs whose expected outputs were produced by
// the Python tokenizers library (normalizers.Precompiled).
func TestPrecompiledRealCharsmap(t *testing.T) {
	steps := realMultilingualNormalizer(t)

	// The charsmap is the first step of the chain
	p, ok := steps[0].(*precompiled)
	if !ok {
		t.Fatalf("expected first normalizer step to be the charsmap, got %T", steps[0])
	}

	testCases := []struct {
		name  string
		input string
		want  string
	}{
		{"fullwidth and ideographic space", "Ｈｅｌｌｏ　ｗｏｒｌｄ", "Hello world"},
		{"ligatures", "ﬁnancial ﬂow", "financial flow"},
		{"circled digits", "①②③", "123"},
		{"roman numeral", "Ⅻ", "XII"},
		// Halfwidth katakana widen and the dakuten stays combining
		// (SentencePiece does not recompose to ガ)
		{"halfwidth katakana with dakuten", "\uFF76\uFF9E\uFF77\uFF9E\uFF78\uFF9E", "\u30AB\u3099\u30AD\u3099\u30AF\u3099"},
		{"no-break space", "a b", "a b"},
		{"unit symbols", "㎞㎏", "kmkg"},
		{"zero width space becomes space", "\u200Bhidden", " hidden"},
		// A two-character grapheme with a whole-grapheme entry:
		// combining acute recomposes onto the base letter
		{"combining accent recomposes", "e\u0301tude", "\u00E9tude"},
		{"precomposed accent unchanged", "\u00E9tude", "\u00E9tude"},
		// Decomposed hangul jamo form one grapheme over the 6-byte
		// whole-grapheme limit and pass through per-character
		{"decomposed hangul jamo", "\u1100\u1161\u11A8", "\u1100\u1161\u11A8"},
		{"precomposed hangul", "\uAC01", "\uAC01"},
		// Inside an over-limit grapheme (emoji ZWJ sequence) each
		// character transforms independently: ZWJ maps to a space
		{"emoji zwj sequence", "\U0001F468\u200D\U0001F469\u200D\U0001F467", "\U0001F468 \U0001F469 \U0001F467"},
		{"multiple combining marks", "x\u0301\u0301y", "x\u0301\u0301y"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := p.normalize(tc.input); got != tc.want {
				t.Errorf("normalize(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// TestRealNormalizerChain runs the multilingual model's full normalizer
// sequence (charsmap, punctuation padding, whitespace collapsing, strip)
// against outputs from the Python tokenizers library.
func TestRealNormalizerChain(t *testing.T) {
	steps := realMultilingualNormalizer(t)

	testCases := []struct {
		input string
		want  string
	}{
		{"  Hello,   world!  ", "Hello , world !"},
		{"don't (stop)", "don ' t ( stop )"},
		{"\u200Bhidden", "hidden"},
		{"Ｈｅｌｌｏ　ｗｏｒｌｄ", "Hello world"},
	}

	for _, tc := range testCases {
		s := tc.input
		for _, step := range steps {
			s = step.normalize(s)
		}
		if s != tc.want {
			t.Errorf("normalize(%q) = %q, want %q", tc.input, s, tc.want)
		}
	}
}
