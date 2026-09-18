package potion

import (
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

type bpeTokenizer struct {
	meta         string // the Metaspace replacement, "▁" for every known model
	prepend      string // Metaspace prepend_scheme: always, first or never
	split        bool   // Metaspace split: start a new word at every meta character
	replaceSpace bool   // the normalizer maps ' ' to meta (affects added-token matching)

	raw        addedTokenTrie // added tokens matched on the raw input
	normalized addedTokenTrie // added tokens matched on the normalized input

	asciiID  [128]int32 // -1 when absent
	bmpID    []int32    // 65536 entries, -1 when absent
	astralID map[rune]int32
	byteID   [256]int32 // <0xXX> byte-fallback tokens, -1 when absent
	metaSyms []uint32   // symbols of a lone meta character
	unkID    int32      // -1 when the model defines none
	fuseUnk  bool
	byteFB   bool

	merges mergeTable

	words      *wordTable
	multi      [][]uint32
	maxWordLen int
	cache      *mergeCache
}

func newBPETokenizer(file tokenizerFile) (*bpeTokenizer, error) {
	model, err := parseBPEModelSection(file.Model)
	if err != nil {
		return nil, err
	}
	if len(model.Vocab) == 0 {
		return nil, fmt.Errorf("bpe model has an empty vocabulary")
	}
	if p := model.Options.ContinuingSubwordPrefix; p != nil && *p != "" {
		return nil, fmt.Errorf("bpe continuing_subword_prefix is not supported")
	}
	if s := model.Options.EndOfWordSuffix; s != nil && *s != "" {
		return nil, fmt.Errorf("bpe end_of_word_suffix is not supported")
	}
	if model.Options.IgnoreMerges {
		return nil, fmt.Errorf("bpe ignore_merges is not supported")
	}
	if model.Options.Dropout != nil && *model.Options.Dropout != 0 {
		return nil, fmt.Errorf("bpe dropout is not supported")
	}

	var pre struct {
		Type          string `json:"type"`
		Replacement   string `json:"replacement"`
		PrependScheme string `json:"prepend_scheme"`
		Split         *bool  `json:"split"`
	}
	if err := json.Unmarshal(file.PreTokenizer, &pre); err != nil {
		return nil, fmt.Errorf("pre_tokenizer section: %w", err)
	}
	if pre.Type != "Metaspace" {
		return nil, fmt.Errorf("unsupported pre-tokenizer for bpe model: %q", pre.Type)
	}
	if _, n := utf8.DecodeRuneInString(pre.Replacement); n != len(pre.Replacement) || n == 0 {
		return nil, fmt.Errorf("metaspace replacement %q is not a single character", pre.Replacement)
	}
	prepend := pre.PrependScheme
	if prepend == "" {
		prepend = "always"
	}
	switch prepend {
	case "always", "first", "never":
	default:
		return nil, fmt.Errorf("unsupported metaspace prepend_scheme %q", prepend)
	}

	t := &bpeTokenizer{
		meta:     pre.Replacement,
		prepend:  prepend,
		split:    pre.Split == nil || *pre.Split,
		bmpID:    make([]int32, 1<<16),
		astralID: make(map[rune]int32),
		unkID:    -1,
		fuseUnk:  model.Options.FuseUnk,
		byteFB:   model.Options.ByteFallback,
	}

	replaceSpace, err := parseBPENormalizer(file.Normalizer, t.meta)
	if err != nil {
		return nil, err
	}
	t.replaceSpace = replaceSpace

	for i := range t.asciiID {
		t.asciiID[i] = -1
	}
	for i := range t.bmpID {
		t.bmpID[i] = -1
	}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for tok, id := range model.Vocab {
		if r, size := utf8.DecodeRuneInString(tok); size == len(tok) && r != utf8.RuneError {
			switch {
			case r < utf8.RuneSelf:
				t.asciiID[r] = id
			case r < 1<<16:
				t.bmpID[r] = id
			default:
				t.astralID[r] = id
			}
		}
	}
	if model.Options.ByteFallback {
		for b := range t.byteID {
			if id, ok := model.Vocab[fmt.Sprintf("<0x%02X>", b)]; ok {
				t.byteID[b] = id
			}
		}
	}
	if model.Options.UnkToken != nil {
		id, ok := model.Vocab[*model.Options.UnkToken]
		if !ok {
			return nil, fmt.Errorf("bpe unk_token %q is not in the vocabulary", *model.Options.UnkToken)
		}
		t.unkID = id
	}
	t.metaSyms = t.appendWordSymbols(nil, t.meta, false)

	rules, err := model.mergeRules()
	if err != nil {
		return nil, err
	}
	t.merges = newMergeTable(len(rules))
	for rank, r := range rules {
		t.merges.insert(uint32(r.a), uint32(r.b), uint32(rank), uint32(r.ab))
	}

	if err := t.addAddedTokens(file.AddedTokens); err != nil {
		return nil, err
	}

	if t.split {
		var whole []string
		for tok := range model.Vocab {
			if !strings.HasPrefix(tok, t.meta) || len(tok) == len(t.meta) {
				continue
			}
			rest := tok[len(t.meta):]
			if strings.IndexByte(rest, ' ') >= 0 || strings.Contains(rest, t.meta) {
				continue
			}
			whole = append(whole, rest)
		}
		results := make([][]uint32, len(whole))
		workers := max(1, min(runtime.GOMAXPROCS(0), 16))
		per := (len(whole) + workers - 1) / workers
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			lo, hi := w*per, min((w+1)*per, len(whole))
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				var syms []uint32
				for i := lo; i < hi; i++ {
					syms = t.merge(t.appendWordSymbols(syms[:0], whole[i], true))
					if len(syms) == 1 {
						results[i] = syms[:1:1]
						syms = nil
					} else {
						results[i] = append([]uint32(nil), syms...)
					}
				}
			}()
		}
		wg.Wait()

		t.words = newWordTable(len(whole))
		for i, rest := range whole {
			if len(results[i]) == 1 {
				t.words.insert(rest, int32(results[i][0]))
			} else {
				t.words.insert(rest, -int32(len(t.multi))-1)
				t.multi = append(t.multi, results[i])
			}
			t.maxWordLen = max(t.maxWordLen, len(rest))
		}
		t.cache = newMergeCache()
	}
	return t, nil
}

func parseBPENormalizer(raw json.RawMessage, meta string) (replaceSpace bool, err error) {
	if len(raw) == 0 || string(raw) == "null" {
		return false, nil
	}
	var cfg struct {
		Type    string `json:"type"`
		Pattern struct {
			String *string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return false, fmt.Errorf("normalizer section: %w", err)
	}
	if cfg.Type == "Replace" && cfg.Pattern.String != nil && *cfg.Pattern.String == " " && cfg.Content == meta {
		return true, nil
	}
	return false, fmt.Errorf("unsupported normalizer for bpe model: %s", raw)
}

func (t *bpeTokenizer) addAddedTokens(raw json.RawMessage) error {
	var entries []struct {
		ID         int32  `json:"id"`
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		Special    bool   `json:"special"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &entries); err != nil {
			return fmt.Errorf("added_tokens section: %w", err)
		}
	}
	for _, e := range entries {
		tok := addedToken{ID: e.ID, SingleWord: e.SingleWord, LStrip: e.LStrip, RStrip: e.RStrip}
		if e.Content == "" {
			continue
		}
		if e.Normalized {
			pattern := e.Content
			if t.replaceSpace {
				pattern = strings.ReplaceAll(pattern, " ", t.meta)
			}
			t.normalized.add(pattern, tok)
		} else {
			t.raw.add(e.Content, tok)
		}
	}
	return nil
}

func (t *bpeTokenizer) tokenize(text string) ([]int, error) {
	ids := make([]int, 0, len(text)/4+1)
	if len(t.raw.tokens) == 0 {
		return t.encodeSegment(ids, text, 0), nil
	}
	start := 0
	for i := 0; i < len(text); {
		if !t.raw.first[text[i]] {
			i++
			continue
		}
		tok, n := t.raw.match(text, i, "")
		if tok < 0 {
			i++
			continue
		}
		s, e := i, i+n
		at := &t.raw.tokens[tok]
		if at.SingleWord && !isolatedWord(text, s, e) {
			i = e
			continue
		}
		if at.LStrip {
			s = max(leftmostSpaceAtEnd(text[:s]), start)
		}
		if at.RStrip {
			e += rightmostSpaceAtStart(text[e:])
		}
		ids = t.encodeSegment(ids, text[start:s], start)
		ids = append(ids, int(at.ID))
		start, i = e, e
	}
	return t.encodeSegment(ids, text[start:], start), nil
}

func (t *bpeTokenizer) encodeSegment(ids []int, seg string, offset int) []int {
	if seg == "" {
		return ids
	}
	if len(t.normalized.tokens) == 0 {
		return t.encodePiece(ids, seg, offset)
	}
	subst := ""
	if t.replaceSpace {
		subst = t.meta
	}
	start := 0
	for i := 0; i < len(seg); {
		b := seg[i]
		if !(t.normalized.first[b] || (b == ' ' && subst != "" && t.normalized.first[subst[0]])) {
			i++
			continue
		}
		tok, n := t.normalized.match(seg, i, subst)
		if tok < 0 {
			i++
			continue
		}
		s, e := i, i+n
		at := &t.normalized.tokens[tok]
		if at.SingleWord && !isolatedWord(seg, s, e) {
			i = e
			continue
		}
		if at.LStrip {
			s = max(leftmostSpaceAtEnd(seg[:s]), start)
		}
		if at.RStrip {
			e += rightmostSpaceAtStart(seg[e:])
		}
		ids = t.encodePiece(ids, seg[start:s], offset+start)
		ids = append(ids, int(at.ID))
		start, i = e, e
	}
	return t.encodePiece(ids, seg[start:], offset+start)
}

func (t *bpeTokenizer) encodePiece(ids []int, piece string, offset int) []int {
	if piece == "" {
		return ids
	}
	prepend := t.prepend == "always" || (t.prepend == "first" && offset == 0)
	if !t.split {
		var stack [128]uint32
		syms := t.appendWordSymbols(stack[:0], piece, prepend && t.boundaryLen(piece) == 0)
		return appendIDs(ids, t.merge(syms))
	}

	j := 0
	withMeta := prepend
	if n := t.boundaryLen(piece); n > 0 {
		withMeta = true
		j = n
	}
	for {
		k := t.nextBoundary(piece, j)
		ids = t.encodeWord(ids, piece[j:k], withMeta)
		if k == len(piece) {
			return ids
		}
		j = k + t.boundaryLen(piece[k:])
		withMeta = true
	}
}

func (t *bpeTokenizer) boundaryLen(s string) int {
	if len(s) == 0 {
		return 0
	}
	if s[0] == ' ' {
		return 1
	}
	if s[0] == t.meta[0] && strings.HasPrefix(s, t.meta) {
		return len(t.meta)
	}
	return 0
}

func (t *bpeTokenizer) nextBoundary(s string, j int) int {
	m0 := t.meta[0]
	for j < len(s) {
		c := s[j]
		if c == ' ' || (c == m0 && strings.HasPrefix(s[j:], t.meta)) {
			return j
		}
		j++
	}
	return len(s)
}

func (t *bpeTokenizer) encodeWord(ids []int, word string, withMeta bool) []int {
	if withMeta && t.words != nil {
		if word == "" {
			return appendIDs(ids, t.metaSyms)
		}
		if len(word) <= t.maxWordLen {
			if v, ok := t.words.lookup(word); ok {
				if v >= 0 {
					return append(ids, int(v))
				}
				return appendIDs(ids, t.multi[-v-1])
			}
		}
		var stack [128]uint32
		if syms, ok := t.cache.get(word, stack[:0]); ok {
			return appendIDs(ids, syms)
		}
		syms := t.merge(t.appendWordSymbols(stack[:0], word, true))
		t.cache.put(word, syms)
		return appendIDs(ids, syms)
	}
	var stack [128]uint32
	syms := t.appendWordSymbols(stack[:0], word, withMeta)
	return appendIDs(ids, t.merge(syms))
}

func appendIDs(ids []int, syms []uint32) []int {
	for _, s := range syms {
		ids = append(ids, int(s))
	}
	return ids
}

func (t *bpeTokenizer) appendWordSymbols(syms []uint32, word string, withMeta bool) []uint32 {
	if withMeta {
		syms = append(syms, t.metaSyms...)
	}
	unk := false
	for i := 0; i < len(word); {
		if b := word[i]; b < utf8.RuneSelf {
			if b == ' ' {
				syms = append(syms, t.metaSyms...)
				unk = false
				i++
				continue
			}
			if id := t.asciiID[b]; id >= 0 {
				syms = append(syms, uint32(id))
				unk = false
				i++
				continue
			}
			syms, unk = t.appendFallback(syms, word[i:i+1], unk)
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(word[i:])
		if r == utf8.RuneError && size == 1 {
			syms, unk = t.appendFallback(syms, word[i:i+1], unk)
			i++
			continue
		}
		var id int32
		if r < 1<<16 {
			id = t.bmpID[r]
		} else if v, ok := t.astralID[r]; ok {
			id = v
		} else {
			id = -1
		}
		if id >= 0 {
			syms = append(syms, uint32(id))
			unk = false
		} else {
			syms, unk = t.appendFallback(syms, word[i:i+size], unk)
		}
		i += size
	}
	return syms
}

func (t *bpeTokenizer) appendFallback(syms []uint32, char string, unk bool) ([]uint32, bool) {
	if t.byteFB {
		ok := true
		for i := 0; i < len(char); i++ {
			if t.byteID[char[i]] < 0 {
				ok = false
				break
			}
		}
		if ok {
			for i := 0; i < len(char); i++ {
				syms = append(syms, uint32(t.byteID[char[i]]))
			}
			return syms, false
		}
	}
	if t.unkID < 0 {
		return syms, false
	}
	if unk && t.fuseUnk {
		return syms, true
	}
	return append(syms, uint32(t.unkID)), true
}

const mergeSmallMax = 128

func (t *bpeTokenizer) merge(syms []uint32) []uint32 {
	n := len(syms)
	if n < 2 {
		return syms
	}
	if n > mergeSmallMax {
		return t.mergeLarge(syms)
	}

	var pairs [mergeSmallMax]uint64
	for i := 0; i < n-1; i++ {
		pairs[i] = t.merges.lookup(syms[i], syms[i+1])
	}
	for n > 1 {
		best, bestVal := -1, noMerge
		for i := 0; i < n-1; i++ {
			if v := pairs[i]; v < bestVal {
				best, bestVal = i, v
			}
		}
		if best < 0 {
			break
		}
		syms[best] = uint32(bestVal)
		copy(syms[best+1:n-1], syms[best+2:n])
		if best < n-2 {
			copy(pairs[best+1:n-2], pairs[best+2:n-1])
		}
		n--
		if best > 0 {
			pairs[best-1] = t.merges.lookup(syms[best-1], syms[best])
		}
		if best < n-1 {
			pairs[best] = t.merges.lookup(syms[best], syms[best+1])
		}
	}
	return syms[:n]
}

func (t *bpeTokenizer) mergeLarge(syms []uint32) []uint32 {
	n := len(syms)
	next := make([]int32, n)
	prev := make([]int32, n)
	for i := range syms {
		next[i] = int32(i + 1)
		prev[i] = int32(i - 1)
	}
	next[n-1] = -1

	heap := make([]uint64, 0, n)
	push := func(pos int32) {
		if v := t.merges.lookup(syms[pos], syms[next[pos]]); v != noMerge {
			heap = heapPush(heap, v&^0xFFFFFFFF|uint64(pos))
		}
	}
	for i := 0; i < n-1; i++ {
		push(int32(i))
	}

	for len(heap) > 0 {
		var top uint64
		heap, top = heapPop(heap)
		pos := int32(top & 0xFFFFFFFF)
		if prev[pos] == -2 || next[pos] < 0 { // removed, or last symbol
			continue
		}
		right := next[pos]
		v := t.merges.lookup(syms[pos], syms[right])
		if v == noMerge || v>>32 != top>>32 {
			continue // stale: the pair at pos changed since it was pushed
		}
		syms[pos] = uint32(v)
		prev[right] = -2
		next[pos] = next[right]
		if next[pos] >= 0 {
			prev[next[pos]] = pos
		}
		if prev[pos] >= 0 {
			push(prev[pos])
		}
		if next[pos] >= 0 {
			push(pos)
		}
	}

	out := syms[:0]
	for i := int32(0); i >= 0; i = next[i] {
		out = append(out, syms[i])
	}
	return out
}

func heapPush(h []uint64, v uint64) []uint64 {
	h = append(h, v)
	i := len(h) - 1
	for i > 0 {
		p := (i - 1) / 2
		if h[p] <= h[i] {
			break
		}
		h[p], h[i] = h[i], h[p]
		i = p
	}
	return h
}

func heapPop(h []uint64) ([]uint64, uint64) {
	top := h[0]
	last := len(h) - 1
	h[0] = h[last]
	h = h[:last]
	i := 0
	for {
		l, r := 2*i+1, 2*i+2
		m := i
		if l < last && h[l] < h[m] {
			m = l
		}
		if r < last && h[r] < h[m] {
			m = r
		}
		if m == i {
			break
		}
		h[m], h[i] = h[i], h[m]
		i = m
	}
	return h, top
}

func leftmostSpaceAtEnd(s string) int {
	return len(strings.TrimRightFunc(s, unicode.IsSpace))
}

func rightmostSpaceAtStart(s string) int {
	return len(s) - len(strings.TrimLeftFunc(s, unicode.IsSpace))
}

func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r) || unicode.Is(unicode.Pc, r)
}

func isolatedWord(text string, s, e int) bool {
	if s > 0 {
		if r, _ := utf8.DecodeLastRuneInString(text[:s]); isWordRune(r) {
			return false
		}
	}
	if e < len(text) {
		if r, _ := utf8.DecodeRuneInString(text[e:]); isWordRune(r) {
			return false
		}
	}
	return true
}
