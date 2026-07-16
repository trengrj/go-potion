package potion

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/rivo/uniseg"
)

// precompiled replays SentencePiece's compiled normalization rules, ported
// from HuggingFace's spm_precompiled crate. The charsmap blob packs a Darts
// double-array trie mapping input byte sequences to offsets into a pool of
// NUL-terminated replacement strings:
// [u32 LE trie byte length][trie: u32 LE units][replacements].
type precompiled struct {
	trie       []uint32
	normalized string
}

func newPrecompiled(charsmapBase64 string) (*precompiled, error) {
	blob, err := base64.StdEncoding.DecodeString(charsmapBase64)
	if err != nil {
		return nil, fmt.Errorf("precompiled charsmap: %w", err)
	}
	if len(blob) < 4 {
		return nil, fmt.Errorf("precompiled charsmap too short: %d bytes", len(blob))
	}
	trieSize := binary.LittleEndian.Uint32(blob)
	if uint64(4)+uint64(trieSize) > uint64(len(blob)) {
		return nil, fmt.Errorf("precompiled charsmap trie size %d exceeds blob size %d", trieSize, len(blob))
	}
	trie := make([]uint32, trieSize/4)
	for i := range trie {
		trie[i] = binary.LittleEndian.Uint32(blob[4+i*4:])
	}
	return &precompiled{
		trie:       trie,
		normalized: string(blob[4+trieSize:]),
	}, nil
}

// Double-array unit accessors, bit-for-bit equal to Darts (and the
// spm_precompiled crate).

func unitHasLeaf(u uint32) bool { return (u>>8)&1 == 1 }

func unitValue(u uint32) uint32 { return u & ((1 << 31) - 1) }

func unitLabel(u uint32) uint32 { return u & ((1 << 31) | 0xFF) }

func unitOffset(u uint32) uint32 { return (u >> 10) << ((u & (1 << 9)) >> 6) }

// firstPrefixMatch walks the trie and returns the value of the shortest
// prefix of key present in it, i.e. the first result of Darts'
// commonPrefixSearch, which is all the Precompiled transform consults.
func (p *precompiled) firstPrefixMatch(key string) (uint32, bool) {
	nodePos := uint32(0)
	unit := p.trie[0]
	nodePos ^= unitOffset(unit)
	for i := 0; i < len(key); i++ {
		c := key[i]
		if c == 0 {
			break
		}
		nodePos ^= uint32(c)
		if int(nodePos) >= len(p.trie) {
			return 0, false
		}
		unit = p.trie[nodePos]
		if unitLabel(unit) != uint32(c) {
			return 0, false
		}
		nodePos ^= unitOffset(unit)
		if unitHasLeaf(unit) {
			if int(nodePos) >= len(p.trie) {
				return 0, false
			}
			return unitValue(p.trie[nodePos]), true
		}
	}
	return 0, false
}

// transform returns the replacement for chunk, if the charsmap has one. The
// returned string is the NUL-terminated entry starting at the trie value's
// offset into the replacement pool.
func (p *precompiled) transform(chunk string) (string, bool) {
	index, ok := p.firstPrefixMatch(chunk)
	if !ok || int(index) > len(p.normalized) {
		return "", false
	}
	end := strings.IndexByte(p.normalized[index:], 0)
	if end < 0 {
		return p.normalized[index:], true
	}
	return p.normalized[index : int(index)+end], true
}

// normalize applies the charsmap the way HuggingFace does: grapheme cluster
// by grapheme cluster, replacing a whole short cluster if it has an entry
// and otherwise each of its characters independently. This odd scheme (see
// the "Future reader" comment in spm_precompiled) is what SentencePiece
// models were validated against, so it is reproduced exactly.
func (p *precompiled) normalize(original string) string {
	var sb strings.Builder
	sb.Grow(len(original))

	state := -1
	rest := original
	for len(rest) > 0 {
		grapheme, tail, _, newState := uniseg.FirstGraphemeClusterInString(rest, state)
		rest, state = tail, newState

		if len(grapheme) < 6 {
			if norm, ok := p.transform(grapheme); ok {
				sb.WriteString(norm)
				continue
			}
		}
		for i := 0; i < len(grapheme); {
			_, size := utf8.DecodeRuneInString(grapheme[i:])
			part := grapheme[i : i+size]
			if norm, ok := p.transform(part); ok {
				sb.WriteString(norm)
			} else {
				sb.WriteString(part)
			}
			i += size
		}
	}
	return sb.String()
}
