package potion

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"sync"
	"testing"
)

const (
	bpeMetaspaceJSON = `{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}`
	bpeReplaceJSON   = `{"type": "Replace", "pattern": {"String": " "}, "content": "▁"}`
)

func bpeModelJSON(vocab []string, merges [][2]string, options string) string {
	v := make(map[string]int, len(vocab))
	for i, tok := range vocab {
		v[tok] = i
	}
	vj, _ := json.Marshal(v)
	mj, _ := json.Marshal(merges)
	if options != "" {
		options = ", " + options
	}
	return fmt.Sprintf(`{"type": "BPE", "vocab": %s, "merges": %s%s}`, vj, mj, options)
}

func newTestBPE(t *testing.T, model, pre, normalizer, added string) *bpeTokenizer {
	t.Helper()
	file := tokenizerFile{
		Model:        json.RawMessage(model),
		PreTokenizer: json.RawMessage(pre),
	}
	if normalizer != "" {
		file.Normalizer = json.RawMessage(normalizer)
	}
	if added != "" {
		file.AddedTokens = json.RawMessage(added)
	}
	tok, err := newBPETokenizer(file)
	if err != nil {
		t.Fatalf("newBPETokenizer: %v", err)
	}
	return tok
}

var testVocab = []string{
	"<unk>", "▁", "a", "b", "c", "ab", "▁a", "▁ab", "abc", "aa", "<0x41>", "<0xC3>", "▁b",
}

const testOptions = `"unk_token": "<unk>", "fuse_unk": true, "byte_fallback": true`

func testBPE(t *testing.T) *bpeTokenizer {
	t.Helper()
	model := bpeModelJSON(testVocab, [][2]string{{"a", "b"}, {"▁", "a"}, {"▁", "ab"}, {"ab", "c"}, {"a", "a"}}, testOptions)
	return newTestBPE(t, model, bpeMetaspaceJSON, bpeReplaceJSON, "")
}

func checkTokenize(t *testing.T, tok tokenizer, input string, want []int) {
	t.Helper()
	got, err := tok.tokenize(input)
	if err != nil {
		t.Fatalf("tokenize(%q): %v", input, err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokenize(%q) = %v, want %v", input, got, want)
	}
}

func TestBPEMergeOrder(t *testing.T) {
	tok := testBPE(t)
	cases := []struct {
		input string
		want  []int
	}{
		{"ab", []int{7}},
		{"abc", []int{7, 4}},
		{"aaa", []int{6, 9}},
		{"baaa", []int{1, 3, 9, 2}},
		{"abc abc", []int{7, 4, 7, 4}},
		{"a b c", []int{6, 1, 3, 1, 4}},
	}
	for _, c := range cases {
		checkTokenize(t, tok, c.input, c.want)
	}
}

func TestBPEMetaspace(t *testing.T) {
	tok := testBPE(t)
	cases := []struct {
		name  string
		input string
		want  []int
	}{
		{"prepended meta", "a", []int{6}},
		{"empty", "", []int{}},
		{"space only", " ", []int{1}},
		{"consecutive spaces stand alone", "a  b", []int{6, 1, 1, 3}},
		{"leading spaces", "  a", []int{1, 6}},
		{"trailing space", "a ", []int{6, 1}},
		{"literal meta char splits too", "a▁b", []int{6, 1, 3}},
		{"leading literal meta is not doubled", "▁a", []int{6}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			checkTokenize(t, tok, c.input, c.want)
		})
	}
}

func TestBPEPrependSchemes(t *testing.T) {
	model := bpeModelJSON(testVocab, [][2]string{{"▁", "a"}}, testOptions)
	added := `[{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}]`

	never := newTestBPE(t, model, `{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": true}`, bpeReplaceJSON, "")
	checkTokenize(t, never, "a b", []int{2, 1, 3})

	first := newTestBPE(t, model, `{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true}`, bpeReplaceJSON, added)
	checkTokenize(t, first, "a<unk>b", []int{6, 0, 3})
	checkTokenize(t, first, "<unk>b", []int{0, 3})

	always := newTestBPE(t, model, `{"type": "Metaspace", "replacement": "▁"}`, bpeReplaceJSON, added)
	checkTokenize(t, always, "a<unk>b", []int{6, 0, 1, 3})
}

func TestBPENoSplit(t *testing.T) {
	model := bpeModelJSON(testVocab, [][2]string{{"▁", "a"}}, testOptions)
	tok := newTestBPE(t, model, `{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false}`, bpeReplaceJSON, "")
	checkTokenize(t, tok, "a b", []int{6, 1, 3})
	checkTokenize(t, tok, "▁a▁b", []int{6, 1, 3})
}

func TestBPEByteFallbackAndUnknown(t *testing.T) {
	tok := testBPE(t)
	checkTokenize(t, tok, "A", []int{1, 10})
	checkTokenize(t, tok, "é", []int{1, 0})
	checkTokenize(t, tok, "éé", []int{1, 0})
	checkTokenize(t, tok, "éaé", []int{1, 0, 2, 0})
	checkTokenize(t, tok, "\xc3", []int{1, 11})
	checkTokenize(t, tok, "\xff", []int{1, 0})

	noFuse := newTestBPE(t, bpeModelJSON(testVocab, nil, `"unk_token": "<unk>", "fuse_unk": false, "byte_fallback": false`), bpeMetaspaceJSON, bpeReplaceJSON, "")
	checkTokenize(t, noFuse, "éé", []int{1, 0, 0})
	checkTokenize(t, noFuse, "A", []int{1, 0})

	noUnk := newTestBPE(t, bpeModelJSON(testVocab, nil, `"byte_fallback": false`), bpeMetaspaceJSON, bpeReplaceJSON, "")
	checkTokenize(t, noUnk, "éaé", []int{1, 2})
}

func TestBPEAddedTokens(t *testing.T) {
	added := `[
		{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
		{"id": 13, "content": "<m>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false, "special": true},
		{"id": 14, "content": "<r>", "single_word": false, "lstrip": false, "rstrip": true, "normalized": false, "special": false},
		{"id": 15, "content": "<n>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
		{"id": 16, "content": "<nn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
		{"id": 17, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
		{"id": 18, "content": "\n\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
		{"id": 19, "content": "[Q] ", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
		{"id": 20, "content": "<w>", "single_word": true, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
	]`
	model := bpeModelJSON(testVocab, [][2]string{{"▁", "a"}, {"▁", "b"}}, testOptions)
	tok := newTestBPE(t, model, bpeMetaspaceJSON, bpeReplaceJSON, added)

	cases := []struct {
		name  string
		input string
		want  []int
	}{
		{"splits text", "a<n>b", []int{6, 15, 12}},
		{"text after a token gets a meta char", "a<n> b", []int{6, 15, 12}},
		{"trailing space before a token stands alone", "a <n>b", []int{6, 1, 15, 12}},
		{"longest match wins", "a<nn>", []int{6, 16}},
		{"lstrip absorbs preceding whitespace", "a  <m>b", []int{6, 13, 12}},
		{"lstrip stops at the previous token", "<n> <m>", []int{15, 13}},
		{"rstrip absorbs following whitespace", "<r>  a", []int{14, 6}},
		{"newline runs", "a\n\n\nb", []int{6, 18, 17, 12}},
		{"normalized token matches a space", "[Q] a", []int{19, 6}},
		{"normalized token matches a literal meta char", "[Q]▁a", []int{19, 6}},
		{"normalized token after raw text", "b [Q] a", []int{12, 1, 19, 6}},
		{"special tokens match too", "a<unk>", []int{6, 0}},
		{"single_word rejects glued matches", "a<w>", []int{6, 0}},
		{"single_word accepts isolated matches", "a <w>", []int{6, 1, 20}},
		{"adjacent tokens", "<n><n>", []int{15, 15}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			checkTokenize(t, tok, c.input, c.want)
		})
	}
}

func TestBPEMergePaths(t *testing.T) {
	vocab := []string{"<unk>", "▁"}
	var grow func(s string)
	grow = func(s string) {
		if len(s) == 4 {
			return
		}
		for _, c := range "abc" {
			vocab = append(vocab, s+string(c))
			grow(s + string(c))
		}
	}
	grow("")
	inVocab := make(map[string]bool, len(vocab))
	for _, v := range vocab {
		inVocab[v] = true
	}
	rng := rand.New(rand.NewSource(7))
	var merges [][2]string
	seen := map[[2]string]bool{}
	for len(merges) < 60 {
		a, b := vocab[2+rng.Intn(len(vocab)-2)], vocab[2+rng.Intn(len(vocab)-2)]
		pair := [2]string{a, b}
		if !inVocab[a+b] || seen[pair] {
			continue
		}
		seen[pair] = true
		merges = append(merges, pair)
	}
	tok := newTestBPE(t, bpeModelJSON(vocab, merges, testOptions), bpeMetaspaceJSON, bpeReplaceJSON, "")

	for i := 0; i < 500; i++ {
		n := 2 + rng.Intn(mergeSmallMax-1)
		var sb strings.Builder
		for j := 0; j < n; j++ {
			sb.WriteByte("abc"[rng.Intn(3)])
		}
		word := sb.String()
		syms := tok.appendWordSymbols(nil, word, true)
		small := tok.merge(append([]uint32(nil), syms...))
		large := tok.mergeLarge(append([]uint32(nil), syms...))
		if !reflect.DeepEqual(small, large) {
			t.Fatalf("merge paths disagree on %q: small %v, large %v", word, small, large)
		}
	}
}

func TestBPECacheConcurrency(t *testing.T) {
	tok := testBPE(t)
	rng := rand.New(rand.NewSource(11))
	words := make([]string, 2000)
	for i := range words {
		var sb strings.Builder
		for j, n := 0, 1+rng.Intn(10); j < n; j++ {
			sb.WriteByte("abcA"[rng.Intn(4)])
		}
		words[i] = sb.String()
	}
	want := make([][]int, len(words))
	for i, w := range words {
		want[i], _ = tok.tokenize(w)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20000; i++ {
				j := (i*7 + g*13) % len(words)
				got, err := tok.tokenize(words[j])
				if err != nil || !reflect.DeepEqual(got, want[j]) {
					t.Errorf("tokenize(%q) = %v, %v; want %v", words[j], got, err, want[j])
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestBPEConfigErrors(t *testing.T) {
	model := bpeModelJSON(testVocab, [][2]string{{"a", "b"}}, testOptions)
	cases := []struct {
		name       string
		model      string
		pre        string
		normalizer string
		wantErr    string
	}{
		{"non-metaspace pre-tokenizer", model, `{"type": "Whitespace"}`, "", "unsupported pre-tokenizer"},
		{"unknown prepend scheme", model, `{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "sometimes"}`, "", "prepend_scheme"},
		{"multi-character replacement", model, `{"type": "Metaspace", "replacement": "▁▁"}`, "", "single character"},
		{"unsupported normalizer", model, bpeMetaspaceJSON, `{"type": "NFKC"}`, "unsupported normalizer"},
		{"ignore_merges", bpeModelJSON(testVocab, nil, `"ignore_merges": true`), bpeMetaspaceJSON, "", "ignore_merges"},
		{"subword prefix", bpeModelJSON(testVocab, nil, `"continuing_subword_prefix": "##"`), bpeMetaspaceJSON, "", "continuing_subword_prefix"},
		{"merge outside vocabulary", bpeModelJSON(testVocab, [][2]string{{"a", "c"}}, ""), bpeMetaspaceJSON, "", "not in the vocabulary"},
		{"unk token outside vocabulary", bpeModelJSON(testVocab, nil, `"unk_token": "<nope>"`), bpeMetaspaceJSON, "", "unk_token"},
		{"empty vocabulary", `{"type": "BPE", "vocab": {}, "merges": []}`, bpeMetaspaceJSON, "", "empty vocabulary"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			file := tokenizerFile{Model: json.RawMessage(c.model), PreTokenizer: json.RawMessage(c.pre)}
			if c.normalizer != "" {
				file.Normalizer = json.RawMessage(c.normalizer)
			}
			_, err := newBPETokenizer(file)
			if err == nil || !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("expected error containing %q, got %v", c.wantErr, err)
			}
		})
	}
}

func TestBPELegacyMergeFormat(t *testing.T) {
	model := `{"type": "BPE", "vocab": {"<unk>": 0, "▁": 1, "a": 2, "b": 3, "ab": 4, "▁ab": 5}, "merges": ["a b", "▁ ab"], "unk_token": "<unk>"}`
	tok := newTestBPE(t, model, bpeMetaspaceJSON, "", "")
	checkTokenize(t, tok, "ab", []int{5})
}

func TestJSONScanner(t *testing.T) {
	doc := []byte(`{
		"version": "1.0",
		"normalizer": null,
		"pre_tokenizer": {"type": "Metaspace", "replacement": "▁"},
		"model": {"type": "BPE", "vocab": {"a\"b": 1, "\n": 2, "x": -3}, "merges": [["a", "b"], "c d"], "unk_token": "<unk>", "dropout": null},
		"added_tokens": [ ],
		"post_processor": {"type": "TemplateProcessing", "single": [], "special_tokens": {}}
	}`)
	file, err := splitTokenizerFile(doc)
	if err != nil {
		t.Fatalf("splitTokenizerFile: %v", err)
	}
	if string(file.Normalizer) != "null" || string(file.AddedTokens) != "[ ]" {
		t.Errorf("unexpected sections: normalizer %q, added_tokens %q", file.Normalizer, file.AddedTokens)
	}
	var pre struct{ Replacement string }
	if err := json.Unmarshal(file.PreTokenizer, &pre); err != nil || pre.Replacement != "▁" {
		t.Errorf("pre_tokenizer section %q: %v", file.PreTokenizer, err)
	}
	typ, err := modelSectionType(file.Model)
	if err != nil || typ != "BPE" {
		t.Errorf("modelSectionType = %q, %v", typ, err)
	}

	m, err := parseBPEModelSection(file.Model)
	if err != nil {
		t.Fatalf("parseBPEModelSection: %v", err)
	}
	if want := map[string]int32{`a"b`: 1, "\n": 2, "x": -3}; !reflect.DeepEqual(m.Vocab, want) {
		t.Errorf("vocab = %v, want %v", m.Vocab, want)
	}
	if m.Options.UnkToken == nil || *m.Options.UnkToken != "<unk>" || m.Options.Dropout != nil {
		t.Errorf("options = %+v", m.Options)
	}
	_, err = m.mergeRules()
	if err == nil || !strings.Contains(err.Error(), `"a" is not in the vocabulary`) {
		t.Errorf("expected vocabulary error, got %v", err)
	}
	m.Vocab = map[string]int32{"a": 0, "b": 1, "ab": 2, "c": 3, "d": 4, "cd": 5}
	rules, err := m.mergeRules()
	if err != nil {
		t.Fatalf("mergeRules: %v", err)
	}
	if want := []mergeRule{{0, 1, 2}, {3, 4, 5}}; !reflect.DeepEqual(rules, want) {
		t.Errorf("rules = %v, want %v", rules, want)
	}

	for _, bad := range []string{`{"model": }`, `{"model": [1, 2}`, `{"model": "unterminated}`, `[]`, `{} x`} {
		if _, err := splitTokenizerFile([]byte(bad)); err == nil {
			t.Errorf("expected error for %s", bad)
		}
	}
	if _, err := parseBPEModelSection([]byte(`{"vocab": {"a": 1.5}}`)); err == nil {
		t.Error("expected error for a fractional vocabulary ID")
	}
}

func TestParsePostProcessor(t *testing.T) {
	template, err := parsePostProcessor(json.RawMessage(`{
		"type": "TemplateProcessing",
		"single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
		"special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}, "<eos>": {"id": "<eos>", "ids": [1], "tokens": ["<eos>"]}}
	}`))
	if err != nil {
		t.Fatalf("TemplateProcessing: %v", err)
	}
	if want := []templateItem{{id: 2}, {sequence: true}, {id: 1}}; !reflect.DeepEqual(template, want) {
		t.Errorf("template = %v, want %v", template, want)
	}

	template, err = parsePostProcessor(json.RawMessage(`{"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}`))
	if err != nil {
		t.Fatalf("BertProcessing: %v", err)
	}
	if want := []templateItem{{id: 101}, {sequence: true}, {id: 102}}; !reflect.DeepEqual(template, want) {
		t.Errorf("template = %v, want %v", template, want)
	}

	if template, err := parsePostProcessor(nil); err != nil || template != nil {
		t.Errorf("absent post_processor = %v, %v", template, err)
	}
	if _, err := parsePostProcessor(json.RawMessage(`{"type": "RobertaProcessing"}`)); err == nil {
		t.Error("expected error for an unsupported post-processor")
	}
}

func TestTokenizerEncode(t *testing.T) {
	tok := &Tokenizer{tok: testBPE(t), template: []templateItem{{id: 2}, {sequence: true}, {id: 1}}}
	ids, err := tok.Encode("ab")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if want := []int{2, 7, 1}; !reflect.DeepEqual(ids, want) {
		t.Errorf("Encode = %v, want %v", ids, want)
	}
	ids, err = tok.Tokenize("ab")
	if err != nil || !reflect.DeepEqual(ids, []int{7}) {
		t.Errorf("Tokenize = %v, %v", ids, err)
	}

	plain := &Tokenizer{tok: testBPE(t)}
	if ids, err := plain.Encode("ab"); err != nil || !reflect.DeepEqual(ids, []int{7}) {
		t.Errorf("Encode without template = %v, %v", ids, err)
	}
	broken := &Tokenizer{tok: testBPE(t), templateErr: fmt.Errorf("nope")}
	if _, err := broken.Encode("ab"); err == nil {
		t.Error("expected Encode to report the post-processor error")
	}
}
