package potion

// sumRowsGeneric adds the embedding rows of tokens into out: for every
// token t, in order, out[i] += emb[t*len(out)+i]. It is the portable
// implementation of sumRows, which the assembly kernels (NEON on arm64, AVX2
// on amd64) replace where available. Every token must be a valid row index —
// the kernels do not bounds-check.
//
// All implementations add the rows in token order, so they produce
// bit-identical sums: an embedding doesn't depend on the machine it was
// computed on, and matches the row-by-row float32 accumulation of model2vec
// (numpy) and model2vec-rs.
//
// Rows are summed four per pass to cut the load/store traffic on out, which
// bounds this loop. Each row is sliced to len(out) so the compiler can
// prove the indexing in bounds and drop the per-element checks.
func sumRowsGeneric(out, emb []float32, tokens []int) {
	dims := len(out)
	k := 0
	for ; k+3 < len(tokens); k += 4 {
		a := emb[tokens[k]*dims:][:len(out)]
		b := emb[tokens[k+1]*dims:][:len(out)]
		c := emb[tokens[k+2]*dims:][:len(out)]
		d := emb[tokens[k+3]*dims:][:len(out)]
		for i := range out {
			out[i] = out[i] + a[i] + b[i] + c[i] + d[i]
		}
	}
	for ; k < len(tokens); k++ {
		a := emb[tokens[k]*dims:][:len(out)]
		for i := range out {
			out[i] += a[i]
		}
	}
}
