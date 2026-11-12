//go:build !purego

package potion

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

var hasAVX2 = cpu.X86.HasAVX2

//go:noescape
func sumRowsAVX2(out *float32, emb *float32, tokens *int, n int, dims int)

// sumRows adds the embedding rows of tokens into out; see sumRowsGeneric
// for the contract. The kernel needs AVX2 and a row width it can handle.
func sumRows(out, emb []float32, tokens []int) {
	if len(tokens) == 0 {
		return
	}
	if !hasAVX2 || len(out) == 0 || len(out)%16 != 0 {
		sumRowsGeneric(out, emb, tokens)
		return
	}
	sumRowsAVX2(unsafe.SliceData(out), unsafe.SliceData(emb), unsafe.SliceData(tokens), len(tokens), len(out))
}
