package potion

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestSumRowsParity checks that the platform's sumRows kernel and the
// portable implementation both produce exactly the sum a plain row-by-row
// loop does, across row widths (including ones the kernels must hand back
// to the portable code) and token counts that exercise the grouped loops
// and their remainders.
func TestSumRowsParity(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const rows = 97
	for _, dims := range []int{1, 8, 16, 24, 32, 48, 64, 128, 256, 512} {
		emb := make([]float32, rows*dims)
		for i := range emb {
			// Mixed magnitudes, so any reordering of the additions
			// shows up as a rounding difference
			emb[i] = (rng.Float32()*2 - 1) * float32(int(1)<<rng.Intn(20))
		}
		for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100} {
			t.Run(fmt.Sprintf("dims=%d/n=%d", dims, n), func(t *testing.T) {
				tokens := make([]int, n)
				for i := range tokens {
					tokens[i] = rng.Intn(rows)
				}
				// Start from a non-zero accumulator, with guard values
				// either side to catch writes outside out
				buf := make([]float32, dims+2)
				buf[0], buf[dims+1] = 42, 43
				got := buf[1 : dims+1]
				generic := make([]float32, dims)
				want := make([]float32, dims)
				for i := range got {
					got[i] = rng.Float32()
					generic[i] = got[i]
					want[i] = got[i]
				}

				sumRows(got, emb, tokens)
				sumRowsGeneric(generic, emb, tokens)
				for _, token := range tokens {
					for i := range want {
						want[i] += emb[token*dims+i]
					}
				}

				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("sumRows index %d: got %v, want %v", i, got[i], want[i])
					}
					if generic[i] != want[i] {
						t.Fatalf("sumRowsGeneric index %d: got %v, want %v", i, generic[i], want[i])
					}
				}
				if buf[0] != 42 || buf[dims+1] != 43 {
					t.Fatalf("kernel wrote outside out: guards %v, %v", buf[0], buf[dims+1])
				}
			})
		}
	}
}

// BenchmarkPoolBigText measures pooling only (embedding row lookup, mean
// and normalization over pre-tokenized chunks), the counterpart of
// BenchmarkTokenizeBigText.
func BenchmarkPoolBigText(b *testing.B) {
	cacheDir := testCacheDir(b)

	encoder, err := New(b.Context(), benchModel(b))
	if err != nil {
		b.Fatalf("Failed to create encoder: %v", err)
	}

	chunks := bigTextChunks(b, cacheDir)
	tokenized := make([][]int, len(chunks))
	var totalTokens int
	for i, chunk := range chunks {
		tokenized[i], err = encoder.Tokenize(chunk)
		if err != nil {
			b.Fatalf("Failed to tokenize chunk: %v", err)
		}
		totalTokens += len(tokenized[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, tokens := range tokenized {
			encoder.poolMean(tokens)
		}
	}
	b.ReportMetric(float64(totalTokens)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mtokens/s")
}
