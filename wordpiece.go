package potion

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/bits"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
	"unsafe"
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
	// first and cont index the vocabulary entries of up to pieceKeyBytes
	// bytes — nearly all of them — for the hot lookups: first holds every
	// such token under its own text, cont holds the "##"-prefixed
	// continuation pieces under their text without the prefix. Longer
	// pieces are looked up in vocab.
	first pieceTable
	cont  pieceTable
	// singleByte caches the vocab IDs of single-ASCII-byte tokens (-1 when
	// absent), so the isolated punctuation words the pre-tokenizer produces
	// skip the hash lookup.
	singleByte [128]int32
	// class drives tokenizeASCII: the pre-tokenizer class of each byte, or
	// classGeneral for bytes the ASCII fast path can't handle.
	class                [256]byte
	unkID                int // -1 if the vocabulary has no unknown token
	maxInputCharsPerWord int
	normalizer           bertNormalizer
}

// pieceKeyBytes is the longest vocabulary piece a pieceTable can index: the
// piece's bytes are packed into a pair of uint64s.
const pieceKeyBytes = 16

// pieceEntry is a pieceTable slot. n is the length of the piece in bytes,
// which disambiguates keys that differ only in trailing NUL bytes.
type pieceEntry struct {
	lo, hi uint64
	id     int32
	n      int32
}

// pieceTable is an open-addressing hash table from vocabulary pieces of up
// to pieceKeyBytes bytes to their token IDs. A piece is keyed by its bytes
// packed little-endian into two uint64s (see pieceKey), so a lookup is a
// couple of multiplies and a probe or two, with no string hashing or
// comparison — several times cheaper than a map[string]int access, which
// otherwise dominates tokenization.
//
// Probing goes through tags, one byte per slot holding a few hash bits of
// the slot's key (0 when empty), and only reads an entry when its tag
// matches. The tags of a full vocabulary fit in the L1 cache while the
// entries don't, which matters most for misses: greedy WordPiece matching
// makes several failed lookups for every piece it finds, and a miss is
// settled from the tags alone.
type pieceTable struct {
	tags    []uint8      // power-of-two length, at most a quarter full
	entries []pieceEntry // parallel to tags
	shift   uint
}

// newPieceTable builds a table holding pieces, sized for a load factor of
// at most 1/4 so linear probe chains stay short.
func newPieceTable(pieces map[string]int) pieceTable {
	size := 16
	for size < 4*len(pieces) {
		size *= 2
	}
	t := pieceTable{
		tags:    make([]uint8, size),
		entries: make([]pieceEntry, size),
		shift:   uint(64 - bits.TrailingZeros(uint(size))),
	}

	// Insert in token ID order: vocabularies list their pieces roughly
	// from most to least frequent, so the pieces looked up most often get
	// their home slots and are found on the first probe. It also makes
	// the layout (and so the performance) independent of map iteration
	// order.
	ordered := make([]string, 0, len(pieces))
	for piece := range pieces {
		ordered = append(ordered, piece)
	}
	slices.SortFunc(ordered, func(a, b string) int {
		if c := cmp.Compare(pieces[a], pieces[b]); c != 0 {
			return c
		}
		return cmp.Compare(a, b)
	})
	for _, piece := range ordered {
		lo, hi := loadKey(piece, 0), loadKey(piece, 8)
		i, tag := t.slot(lo, hi)
		for t.tags[i] != 0 {
			i = (i + 1) & (size - 1)
		}
		t.tags[i] = tag
		t.entries[i] = pieceEntry{lo: lo, hi: hi, id: int32(pieces[piece]), n: int32(len(piece))}
	}
	return t
}

// slot returns the home slot of a key (Fibonacci hashing) and its non-zero
// tag, taken from the hash bits just below the ones that pick the slot.
func (t *pieceTable) slot(lo, hi uint64) (int, uint8) {
	h := (lo ^ hi*0xC2B2AE3D27D4EB4F) * 0x9E3779B97F4A7C15
	return int(h >> t.shift), uint8(h>>(t.shift-7)) | 0x80
}

// lookup returns the token ID of the n-byte piece with key (lo, hi), or -1.
func (t *pieceTable) lookup(lo, hi uint64, n int) int {
	i, tag := t.slot(lo, hi)
	for ; ; i++ {
		i &= len(t.tags) - 1
		switch t.tags[i] {
		case tag:
			if e := &t.entries[i]; e.lo == lo && e.hi == hi && int(e.n) == n {
				return int(e.id)
			}
		case 0:
			return -1
		}
	}
}

// keyMask[n] keeps the low n bytes of a key half.
var keyMask = [9]uint64{
	0, 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF,
	0xFFFFFFFFFF, 0xFFFFFFFFFFFF, 0xFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
}

// pieceKey returns the key of the first n <= pieceKeyBytes bytes of the
// text whose first sixteen bytes are packed into lo and hi.
func pieceKey(lo, hi uint64, n int) (uint64, uint64) {
	if n <= 8 {
		return lo & keyMask[n], 0
	}
	return lo, hi & keyMask[n-8]
}

// loadKey packs up to eight bytes of s starting at i into a uint64,
// little-endian, zero-padded when s ends first (i may be past the end).
// Two of them, eight bytes apart, make the key of a piece that fills them;
// pieceKey cuts shorter pieces out of that.
func loadKey(s string, i int) uint64 {
	if i+8 <= len(s) {
		return binary.LittleEndian.Uint64(unsafe.Slice(unsafe.StringData(s), len(s))[i:])
	}
	return loadKeyTail(s, i)
}

// loadKeyTail is loadKey for the last few bytes of s, kept out of line so
// loadKey itself inlines.
func loadKeyTail(s string, i int) uint64 {
	var key uint64
	for j := len(s) - 1; j >= i; j-- {
		key = key<<8 | uint64(s[j])
	}
	return key
}

// scanWord classifies eight packed bytes at once. It returns the number of
// leading bytes that are ASCII letters or digits — the bulk of what words
// are made of — and the bytes themselves with ASCII letters lowercased.
//
// Per byte, with the high bit masked off so sums can't carry into the next
// byte: adding 0x80-lo sets the high bit when the byte is >= lo, and adding
// 0x80-hi-1 sets it when the byte is > hi, so (ge &^ gt) flags the range
// [lo, hi]. Letters are tested with bit 0x20 forced on, which folds A-Z
// onto a-z and moves no other byte into that range.
func scanWord(x uint64) (n int, lowered uint64) {
	const (
		ones = 0x0101010101010101
		high = ones * 0x80
	)
	low7 := x &^ high
	folded := low7 | ones*0x20
	letters := (folded + ones*(0x80-'a')) &^ (folded + ones*(0x80-'z'-1))
	digits := (low7 + ones*(0x80-'0')) &^ (low7 + ones*(0x80-'9'-1))
	// High bit set in every byte that is not an ASCII letter or digit
	other := ^((letters | digits) &^ x) & high
	return bits.TrailingZeros64(other) >> 3, x | (letters&^x&high)>>2
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

	return newWordPiece(cfg.Model.Vocab, unkID, maxInputCharsPerWord, *bertNormalizerFromConfig(cfg)), nil
}

// newWordPiece assembles a wordPieceTokenizer and its lookup tables from a
// vocabulary and normalizer settings.
func newWordPiece(vocab map[string]int, unkID, maxInputCharsPerWord int, normalizer bertNormalizer) *wordPieceTokenizer {
	first := make(map[string]int)
	cont := make(map[string]int)
	for piece, id := range vocab {
		if len(piece) > 0 && len(piece) <= pieceKeyBytes {
			first[piece] = id
		}
		if rest, ok := strings.CutPrefix(piece, "##"); ok && len(rest) > 0 && len(rest) <= pieceKeyBytes {
			cont[rest] = id
		}
	}

	t := &wordPieceTokenizer{
		vocab:                vocab,
		first:                newPieceTable(first),
		cont:                 newPieceTable(cont),
		singleByte:           singleByteIDs(vocab),
		unkID:                unkID,
		maxInputCharsPerWord: maxInputCharsPerWord,
		normalizer:           normalizer,
	}
	for b := range t.class {
		switch {
		case maxInputCharsPerWord < pieceKeyBytes:
			// tokenizeASCII relies on words that fit a pieceTable key
			// being within the length limit; no real tokenizer has a
			// limit this low, so just route everything to word2tok
			t.class[b] = classGeneral
		case b >= utf8.RuneSelf:
			t.class[b] = classGeneral
		case normalizer.CleanText && (b < 0x20 || b == 0x7f) && b != '\t' && b != '\n' && b != '\r':
			// Control characters are deleted by the normalizer, which
			// can join the text either side of them into one word
			t.class[b] = classGeneral
		default:
			t.class[b] = asciiClass[b]
		}
	}
	return t
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
	// classGeneral marks, in wordPieceTokenizer.class only, the bytes that
	// send tokenizeASCII back to the general pipeline
	classGeneral
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

// word2tok appends the WordPiece token IDs of a single normalized word to
// dst using greedy longest-match-first, and returns the extended slice. A
// word that cannot be fully tokenized (or exceeds maxInputCharsPerWord) maps
// to the unknown token as a whole in the reference implementation; model2vec
// then drops unknown tokens since their embedding carries no signal, so such
// words contribute nothing to dst.
func (t *wordPieceTokenizer) word2tok(dst []int, word string) ([]int, error) {
	mark := len(dst)

	// Rune count never exceeds byte length, so words short enough in bytes
	// skip the counting scan entirely
	if len(word) > t.maxInputCharsPerWord && utf8.RuneCountInString(word) > t.maxInputCharsPerWord {
		return t.dropUnknown(dst, mark, word)
	}

	// Scratch buffer for "##"-prefixed lookups; the vocab access via
	// string(buf) does not allocate
	var scratch [512]byte

	start := 0
	for start < len(word) {
		// Candidates of up to pieceKeyBytes bytes are all prefixes of
		// the same sixteen bytes, so load them once per piece
		lo, hi := loadKey(word, start), loadKey(word, start+8)
		table := &t.first
		if start > 0 {
			table = &t.cont
		}

		end := len(word)
		token := -1
		for end > start {
			if n := end - start; n <= pieceKeyBytes {
				l, h := pieceKey(lo, hi, n)
				token = table.lookup(l, h, n)
			} else if start == 0 {
				if id, ok := t.vocab[word[:end]]; ok {
					token = id
				}
			} else {
				buf := append(append(scratch[:0], '#', '#'), word[start:end]...)
				if id, ok := t.vocab[string(buf)]; ok {
					token = id
				}
			}
			if token >= 0 {
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
	// Cloning keeps word from escaping, so callers can pass stack-backed
	// strings
	return nil, fmt.Errorf("cannot tokenize word and no unknown token available: %s", strings.Clone(word))
}

// tokenizeASCII is the fast path for text made only of ASCII bytes the
// normalizer keeps, which is most text the BERT-style models see. It fuses
// normalization, pre-tokenization and vocabulary lookup into one pass over
// the raw sentence with no intermediate strings. A word is measured and
// lowercased eight bytes at a time (scanWord) rather than byte by byte, and
// when it is up to pieceKeyBytes long — all but a fraction of a percent of
// English words — those same bytes are its lookup key, so it resolves with
// a single pieceTable probe or, if it needs splitting, through pieces2tok
// without ever leaving registers. Only longer words go through word2tok.
//
// It appends to dst and reports ok = false, with dst as it was, as soon as
// it meets a byte it can't handle (non-ASCII, or a control
// character the normalizer deletes); the caller then runs the general
// pipeline on the whole sentence.
func (t *wordPieceTokenizer) tokenizeASCII(dst []int, s string) (tokens []int, ok bool, err error) {
	mark := len(dst)
	lowercase := t.normalizer.Lowercase
	for i := 0; i < len(s); {
		b := s[i]
		switch t.class[b] {
		case asciiSpace:
			i++
			continue
		case asciiPunct:
			id := t.singleByte[b]
			if id < 0 {
				if dst, err = t.dropUnknown(dst, len(dst), s[i:i+1]); err != nil {
					return nil, false, err
				}
			} else {
				dst = append(dst, int(id))
			}
			i++
			continue
		case asciiWord:
		default:
			return dst[:mark], false, nil
		}

		// Measure the run of letters and digits at i, up to sixteen of
		// them. loadKey zero-pads at the end of s, which stops the count
		// there.
		head := loadKey(s, i)
		n, lo := scanWord(head)
		if !lowercase {
			lo = head
		}
		var hi uint64
		if n == 8 {
			tail := loadKey(s, i+8)
			m, lowered := scanWord(tail)
			if !lowercase {
				lowered = tail
			}
			n += m
			hi = lowered & keyMask[m]
		} else {
			lo &= keyMask[n]
		}

		// The run is a whole word when the sentence, a space or a
		// punctuation character ends it. Anything else continues the
		// word or falls back, which the byte loop below sorts out.
		end := i + n
		if end == len(s) || t.class[s[end]]-asciiSpace <= asciiPunct-asciiSpace {
			if id := t.first.lookup(lo, hi, n); id >= 0 {
				dst = append(dst, id)
				i = end
				continue
			}
			if dst, ok = t.pieces2tok(dst, lo, hi, n); ok {
				i = end
				continue
			}
			// Unknown word: word2tok below drops or reports it
		}

		for end < len(s) && t.class[s[end]] == asciiWord {
			end++
		}
		if end < len(s) && t.class[s[end]] == classGeneral {
			// The word may continue past a deleted control character or
			// into non-ASCII text
			return dst[:mark], false, nil
		}
		if dst, err = t.word2tokASCII(dst, s[i:end]); err != nil {
			return nil, false, err
		}
		i = end
	}
	return dst, true, nil
}

// pieces2tok is word2tok for a normalized all-ASCII word of n <=
// pieceKeyBytes bytes packed into lo and hi, already known not to be a
// vocabulary entry as a whole. Every candidate piece of such a word is a
// byte range of its key, so the greedy longest-match search needs only
// shifts, masks and pieceTable probes. It reports false, leaving dst as it
// was, when the word can't be split into vocabulary pieces.
func (t *wordPieceTokenizer) pieces2tok(dst []int, lo, hi uint64, n int) ([]int, bool) {
	mark := len(dst)
	table := &t.first
	for end := n - 1; n > 0; end = n {
		id := -1
		for ; end > 0; end-- {
			l, h := pieceKey(lo, hi, end)
			if id = table.lookup(l, h, end); id >= 0 {
				break
			}
		}
		if id < 0 {
			return dst[:mark], false
		}
		dst = append(dst, id)

		// Shift the matched bytes out of the key
		if end >= 8 {
			lo, hi = hi>>(8*uint(end-8)), 0
		} else {
			lo, hi = lo>>(8*uint(end))|hi<<(64-8*uint(end)), hi>>(8*uint(end))
		}
		n -= end
		table = &t.cont
	}
	return dst, true
}

// word2tokASCII is word2tok for a raw (not yet normalized) all-ASCII word:
// lowercasing is the only normalization left to apply, done into a stack
// buffer for all but pathologically long words.
func (t *wordPieceTokenizer) word2tokASCII(dst []int, word string) ([]int, error) {
	if !t.normalizer.Lowercase {
		return t.word2tok(dst, word)
	}
	var scratch [128]byte
	buf := scratch[:0]
	if len(word) > len(scratch) {
		buf = make([]byte, 0, len(word))
	}
	for i := 0; i < len(word); i++ {
		b := word[i]
		if b >= 'A' && b <= 'Z' {
			b += 'a' - 'A'
		}
		buf = append(buf, b)
	}
	return t.word2tok(dst, unsafe.String(unsafe.SliceData(buf), len(buf)))
}

func (t *wordPieceTokenizer) tokenize(sentence string) ([]int, error) {
	// English text averages a little under one token per four bytes
	allTokens := make([]int, 0, len(sentence)/4+8)
	allTokens, ok, err := t.tokenizeASCII(allTokens, sentence)
	if ok || err != nil {
		return allTokens, err
	}
	return t.tokenizeGeneral(allTokens, sentence)
}

// tokenizeGeneral runs the full pipeline — normalizer, pre-tokenizer, then
// WordPiece per word — for any input, appending to dst.
func (t *wordPieceTokenizer) tokenizeGeneral(dst []int, sentence string) ([]int, error) {
	normalized := t.normalizer.normalize(sentence)
	words := preTokenize(normalized)

	var err error
	for _, word := range words {
		dst, err = t.word2tok(dst, word)
		if err != nil {
			return nil, err
		}
	}

	return dst, nil
}
