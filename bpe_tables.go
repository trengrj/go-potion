package potion

import (
	"encoding/binary"
	"hash/maphash"
	"sync/atomic"
)

type mergeTable struct {
	keys        []uint64 // symbol pair, or emptyKey
	vals        []uint64 // rank<<32 | merged ID
	mask        uint64
	shift       uint
	filter      []uint64
	filterShift uint
}

const (
	emptyKey = ^uint64(0)
	noMerge  = ^uint64(0)
)

func newMergeTable(n int) mergeTable {
	bits := uint(8) // at most 2/3 full; tiny tables stay simple
	for 1<<bits < n*3/2 {
		bits++
	}
	filterBits := bits + 3 // at least 12 filter bits per pair
	m := mergeTable{
		keys:        make([]uint64, 1<<bits),
		vals:        make([]uint64, 1<<bits),
		mask:        1<<bits - 1,
		shift:       64 - bits,
		filter:      make([]uint64, 1<<(filterBits-6)),
		filterShift: 64 - (filterBits - 6),
	}
	for i := range m.keys {
		m.keys[i] = emptyKey
	}
	return m
}

func pairHash(key uint64) (h uint64, f uint64, bits uint64) {
	h = key * 0x9E3779B97F4A7C15
	f = (key ^ h>>32) * 0xD6E8FEB86659FD93
	bits = 1<<(f&63) | 1<<(f>>6&63)
	return h, f, bits
}

func (m *mergeTable) insert(a, b uint32, rank, id uint32) {
	key := uint64(a)<<32 | uint64(b)
	h, f, bits := pairHash(key)
	m.filter[f>>m.filterShift] |= bits
	i := h >> m.shift
	for m.keys[i] != emptyKey && m.keys[i] != key {
		i = (i + 1) & m.mask
	}
	m.keys[i] = key
	m.vals[i] = uint64(rank)<<32 | uint64(id)
}

func (m *mergeTable) lookup(a, b uint32) uint64 {
	key := uint64(a)<<32 | uint64(b)
	h, f, bits := pairHash(key)
	if m.filter[f>>m.filterShift]&bits != bits {
		return noMerge
	}
	i := h >> m.shift
	for {
		k := m.keys[i]
		if k == key {
			return m.vals[i]
		}
		if k == emptyKey {
			return noMerge
		}
		i = (i + 1) & m.mask
	}
}

type wordTable struct {
	slots []wordSlot
	mask  uint64
	seed  maphash.Seed
	pool  []byte
	count int
}

const wordInline = 24

type wordSlot struct {
	tag uint32 // fingerprint<<8 | byte length; 0 marks an empty slot
	val int32
	key [wordInline]byte // the word, or its pool offset when longer
}

func newWordTable(capacity int) *wordTable {
	bits := uint(4)
	for 1<<bits < capacity*2 {
		bits++
	}
	return &wordTable{
		slots: make([]wordSlot, 1<<bits),
		mask:  1<<bits - 1,
		seed:  maphash.MakeSeed(),
	}
}

func (w *wordTable) hash(word string) (h uint64, tag uint32) {
	h = maphash.String(w.seed, word)
	return h, uint32(h>>40)<<8 | uint32(len(word))
}

func (w *wordTable) insert(word string, val int32) {
	if len(word) == 0 || len(word) > 255 {
		return
	}
	h, tag := w.hash(word)
	i := h & w.mask
	for w.slots[i].tag != 0 {
		i = (i + 1) & w.mask
	}
	s := &w.slots[i]
	s.tag, s.val = tag, val
	if len(word) <= wordInline {
		copy(s.key[:], word)
	} else {
		binary.LittleEndian.PutUint32(s.key[:], uint32(len(w.pool)))
		w.pool = append(w.pool, word...)
	}
	w.count++
}

func (w *wordTable) lookup(word string) (int32, bool) {
	if len(word) > 255 {
		return 0, false
	}
	h, tag := w.hash(word)
	i := h & w.mask
	for {
		s := &w.slots[i]
		if s.tag == 0 {
			return 0, false
		}
		if s.tag == tag {
			if len(word) <= wordInline {
				if string(s.key[:len(word)]) == word {
					return s.val, true
				}
			} else {
				off := binary.LittleEndian.Uint32(s.key[:])
				if string(w.pool[off:off+uint32(len(word))]) == word {
					return s.val, true
				}
			}
		}
		i = (i + 1) & w.mask
	}
}

type mergeCache struct {
	slots []cacheSlot
	mask  uint64
	seed  maphash.Seed
}

type cacheSlot struct {
	seq atomic.Uint64
	key [3]atomic.Uint64 // the word's bytes, zero padded
	res [2]atomic.Uint64 // token IDs, cacheIDBits each; count in the top 4 bits of res[1]
	_   [2]uint64        // pad to a cache line
}

const (
	cacheWordBytes = 24
	cacheMaxTokens = 6
	cacheIDBits    = 20
	cacheIDMask    = 1<<cacheIDBits - 1
	cacheSlotBits  = 17 // 131072 slots of 64 bytes: 8 MB
)

func newMergeCache() *mergeCache {
	return &mergeCache{
		slots: make([]cacheSlot, 1<<cacheSlotBits),
		mask:  1<<cacheSlotBits - 1,
		seed:  maphash.MakeSeed(),
	}
}

func packWord(word string) (k [3]uint64) {
	var buf [cacheWordBytes]byte
	copy(buf[:], word)
	for i := range k {
		k[i] = binary.LittleEndian.Uint64(buf[i*8:])
	}
	return k
}

func (c *mergeCache) get(word string, syms []uint32) ([]uint32, bool) {
	if len(word) > cacheWordBytes {
		return syms, false
	}
	set := maphash.String(c.seed, word) & c.mask &^ 1
	w := packWord(word)
	for way := uint64(0); way < 2; way++ {
		s := &c.slots[set|way]
		v1 := s.seq.Load()
		if v1&1 != 0 || int(v1>>32&0xFF) != len(word) {
			continue
		}
		k0, k1, k2 := s.key[0].Load(), s.key[1].Load(), s.key[2].Load()
		r0, r1 := s.res[0].Load(), s.res[1].Load()
		if s.seq.Load() != v1 || k0 != w[0] || k1 != w[1] || k2 != w[2] {
			continue
		}
		n := int(r1 >> 60)
		for i := 0; i < n; i++ {
			if i < 3 {
				syms = append(syms, uint32(r0>>(i*cacheIDBits))&cacheIDMask)
			} else {
				syms = append(syms, uint32(r1>>((i-3)*cacheIDBits))&cacheIDMask)
			}
		}
		return syms, true
	}
	return syms, false
}

func (c *mergeCache) put(word string, syms []uint32) {
	if len(word) > cacheWordBytes || len(syms) > cacheMaxTokens {
		return
	}
	var r [2]uint64
	for i, id := range syms {
		if id > cacheIDMask {
			return
		}
		if i < 3 {
			r[0] |= uint64(id) << (i * cacheIDBits)
		} else {
			r[1] |= uint64(id) << ((i - 3) * cacheIDBits)
		}
	}
	r[1] |= uint64(len(syms)) << 60

	h := maphash.String(c.seed, word)
	set := h & c.mask &^ 1
	way := h >> 63
	if c.slots[set|way].seq.Load() != 0 && c.slots[set|way^1].seq.Load() == 0 {
		way ^= 1
	}
	s := &c.slots[set|way]
	old := s.seq.Load()
	if old&1 != 0 || !s.seq.CompareAndSwap(old, old|1) {
		return
	}
	k := packWord(word)
	for i := range k {
		s.key[i].Store(k[i])
	}
	s.res[0].Store(r[0])
	s.res[1].Store(r[1])
	s.seq.Store((old+2)&0xFFFFFFFF | uint64(len(word))<<32)
}

type addedToken struct {
	ID         int32
	SingleWord bool // match only when not glued to word characters
	LStrip     bool // the match absorbs whitespace before it
	RStrip     bool // the match absorbs whitespace after it
}

type addedTokenTrie struct {
	first  [256]bool
	nodes  []trieNode
	tokens []addedToken
}

type trieNode struct {
	edges []trieEdge
	token int32 // index into tokens, -1 for none
}

type trieEdge struct {
	b    byte
	node int32
}

func (t *addedTokenTrie) add(pattern string, tok addedToken) {
	if len(t.nodes) == 0 {
		t.nodes = append(t.nodes, trieNode{token: -1})
	}
	if pattern == "" {
		return
	}
	t.first[pattern[0]] = true
	node := int32(0)
	for i := 0; i < len(pattern); i++ {
		node = t.child(node, pattern[i])
	}
	t.nodes[node].token = int32(len(t.tokens))
	t.tokens = append(t.tokens, tok)
}

func (t *addedTokenTrie) child(node int32, b byte) int32 {
	for _, e := range t.nodes[node].edges {
		if e.b == b {
			return e.node
		}
	}
	next := int32(len(t.nodes))
	t.nodes = append(t.nodes, trieNode{token: -1})
	t.nodes[node].edges = append(t.nodes[node].edges, trieEdge{b: b, node: next})
	return next
}

func (t *addedTokenTrie) step(node int32, b byte) int32 {
	for _, e := range t.nodes[node].edges {
		if e.b == b {
			return e.node
		}
	}
	return -1
}

func (t *addedTokenTrie) match(text string, i int, subst string) (tok int32, n int) {
	tok = -1
	node := int32(0)
	for j := i; j < len(text); j++ {
		b := text[j]
		if b == ' ' && subst != "" {
			for k := 0; k < len(subst) && node >= 0; k++ {
				node = t.step(node, subst[k])
			}
		} else {
			node = t.step(node, b)
		}
		if node < 0 {
			break
		}
		if id := t.nodes[node].token; id >= 0 {
			tok, n = id, j+1-i
		}
	}
	return tok, n
}
