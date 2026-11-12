//go:build !purego

package potion

import "unsafe"

//go:noescape
func sumRowsNEON(out *float32, emb *float32, tokens *int, n int, dims int)

// sumRows adds the embedding rows of tokens into out; see sumRowsGeneric
// for the contract. NEON is part of the arm64 baseline, so the kernel needs
// no feature detection, only a row width it can handle.
func sumRows(out, emb []float32, tokens []int) {
	if len(tokens) == 0 {
		return
	}
	if len(out) == 0 || len(out)%16 != 0 {
		sumRowsGeneric(out, emb, tokens)
		return
	}
	sumRowsNEON(unsafe.SliceData(out), unsafe.SliceData(emb), unsafe.SliceData(tokens), len(tokens), len(out))
}
