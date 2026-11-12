//go:build purego || (!arm64 && !amd64)

package potion

// sumRows adds the embedding rows of tokens into out; see sumRowsGeneric
// for the contract.
func sumRows(out, emb []float32, tokens []int) {
	sumRowsGeneric(out, emb, tokens)
}
