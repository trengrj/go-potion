package potion

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/nlpodyssey/safetensors"
	"github.com/tphakala/simd/f32"
)

// Model identifies one of the POTION models the package can load
type Model string

const (
	BASE2M           Model = "BASE2M"
	BASE4M           Model = "BASE4M"
	BASE8M           Model = "BASE8M"
	BASE32M          Model = "BASE32M"
	RETRIEVAL32M     Model = "RETRIEVAL32M"
	SCIENCE32M       Model = "SCIENCE32M"
	CODE16M          Model = "CODE16M"
	CODE16MV2        Model = "CODE16MV2"
	MULTILINGUAL128M Model = "MULTILINGUAL128M"
)

// modelRepos maps each supported model to its HuggingFace repository. This
// covers every model in the POTION collection
// (https://huggingface.co/collections/minishlab/potion).
var modelRepos = map[Model]string{
	BASE2M:           "minishlab/potion-base-2M",
	BASE4M:           "minishlab/potion-base-4M",
	BASE8M:           "minishlab/potion-base-8M",
	BASE32M:          "minishlab/potion-base-32M",
	RETRIEVAL32M:     "minishlab/potion-retrieval-32M",
	SCIENCE32M:       "minishlab/potion-science-32M",
	CODE16M:          "minishlab/potion-code-16M",
	CODE16MV2:        "minishlab/potion-code-16M-v2",
	MULTILINGUAL128M: "minishlab/potion-multilingual-128M",
}

// Models returns all supported models in a stable order.
func Models() []Model {
	return []Model{
		BASE2M, BASE4M, BASE8M, BASE32M,
		RETRIEVAL32M, SCIENCE32M,
		CODE16M, CODE16MV2,
		MULTILINGUAL128M,
	}
}

// tokenizer converts raw text into the token IDs model2vec would produce for
// the same model. The two implementations mirror the two tokenizer families
// used by POTION models: WordPiece (all base/retrieval/science/code models)
// and SentencePiece Unigram (potion-multilingual-128M).
type tokenizer interface {
	tokenize(sentence string) ([]int, error)
}

// tokenizerFile is the top-level structure of tokenizer.json. The normalizer
// and model sections differ per tokenizer family, so they are kept raw and
// interpreted by the family-specific constructors.
type tokenizerFile struct {
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        json.RawMessage `json:"model"`
}

// Potion is the main struct for loading and using the model
type Potion struct {
	embeddings []float32
	dimensions int
	// weights and mapping come from vocabulary-quantized models
	// (potion-code-16M): the embedding row for token t is
	// embeddings[mapping[t]] scaled by weights[t]. Both are nil for
	// ordinary models, in which case token IDs index embeddings directly.
	weights []float64
	mapping []int64
	tok     tokenizer
	// unmap releases the read-only file mapping that embeddings aliases.
	// Nil when the weights live on the Go heap instead (non-unix platforms,
	// F16 models, or misaligned/big-endian fallbacks).
	unmap func() error
}

// l2 calculates the L2 norm of a vector
func l2(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// ensureFile downloads a file from a URL unless it already exists at the
// destination path
func ensureFile(ctx context.Context, url string, destination string) error {
	if _, err := os.Stat(destination); err == nil {
		return nil
	}
	return downloadFile(ctx, url, destination)
}

// downloadFile downloads a URL to a destination path atomically: the body
// is written to a temporary file in the same directory and renamed into
// place, so an interrupted download never leaves a truncated file at the
// final path. This matters because the cache directory may be shared
// across projects. The context cancels both the request and the body copy,
// which matters for model files hundreds of megabytes in size.
func downloadFile(ctx context.Context, url string, destination string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: %s", resp.Status)
	}

	tmp, err := os.CreateTemp(filepath.Dir(destination), filepath.Base(destination)+".tmp-*")
	if err != nil {
		return err
	}
	defer func() {
		tmp.Close()
		os.Remove(tmp.Name()) // no-op once renamed
	}()

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), destination)
}

// tensorAlignment is the boundary the embeddings tensor of cached
// safetensors files is aligned to. Page alignment gives the mmapped tensor
// the same layout a heap-allocated copy would have: rows sit on cache-line
// boundaries (embedding rows are a multiple of 64 bytes for every POTION
// model, so row loads never straddle a cache line) and never cross a page
// boundary either. Cache-line alignment alone measures ~5% slower on
// encode than page alignment.
const tensorAlignment = 4096

// alignSafetensors pads the JSON header of a cached safetensors file so the
// embeddings tensor — the one Encode streams on every call, and the only
// one mmapped zero-copy — starts on a tensorAlignment boundary. HuggingFace
// pads headers to 8 bytes, which leaves the mmapped embeddings 24 bytes off
// a cache line and costs ~10% of encode throughput; trailing spaces after
// the header JSON are the format's own padding mechanism, so the rewritten
// file is still a valid safetensors file. Files downloaded by earlier
// releases of this library are realigned the same way the first time they
// are loaded. The rewrite is atomic (temp file + rename), so a concurrent
// load in another process either sees the old file or the new one, never a
// partial write. Already-aligned files are left untouched.
func alignSafetensors(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var sizeBuf [8]byte
	if _, err := io.ReadFull(f, sizeBuf[:]); err != nil {
		return err
	}
	headerLen := binary.LittleEndian.Uint64(sizeBuf[:])

	info, err := f.Stat()
	if err != nil {
		return err
	}
	if headerLen > uint64(info.Size())-8 {
		return fmt.Errorf("safetensors header length %d exceeds file size %d", headerLen, info.Size())
	}

	header := make([]byte, headerLen)
	if _, err := io.ReadFull(f, header); err != nil {
		return err
	}

	// The embeddings tensor's offset within the data section is fixed, so
	// pad the header until file offset 8+headerLen+embeddingsOffset lands
	// on the alignment boundary. In most POTION models embeddings is the
	// only tensor (offset 0); in the quantized ones it sits after the
	// weights and mapping tensors, which stay heap-copied and don't need
	// aligning.
	var tensors map[string]struct {
		DataOffsets []uint64 `json:"data_offsets"`
	}
	if err := json.Unmarshal(header, &tensors); err != nil {
		return fmt.Errorf("safetensors header: %w", err)
	}
	emb, ok := tensors["embeddings"]
	if !ok || len(emb.DataOffsets) != 2 {
		return fmt.Errorf("embeddings tensor not found in safetensors header")
	}

	pad := (tensorAlignment - (8+headerLen+emb.DataOffsets[0])%tensorAlignment) % tensorAlignment
	if pad == 0 {
		return nil
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	defer func() {
		tmp.Close()
		os.Remove(tmp.Name()) // no-op once renamed
	}()

	binary.LittleEndian.PutUint64(sizeBuf[:], headerLen+pad)
	if _, err := tmp.Write(sizeBuf[:]); err != nil {
		return err
	}
	if _, err := tmp.Write(header); err != nil {
		return err
	}
	spaces := make([]byte, pad)
	for i := range spaces {
		spaces[i] = ' '
	}
	if _, err := tmp.Write(spaces); err != nil {
		return err
	}
	if _, err := io.Copy(tmp, f); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// float16ToFloat32 converts an IEEE 754 half-precision value (as raw bits)
// to float32. The conversion is exact: every float16 value is representable
// as a float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign) // +/- zero
		}
		// Subnormal half: renormalize into a normal float32
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3FF
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1F:
		return math.Float32frombits(sign | 0xFF<<23 | mant<<13) // Inf / NaN
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

// hostLittleEndian reports whether this machine's byte order matches the
// little-endian layout safetensors files are stored in.
var hostLittleEndian = binary.NativeEndian.Uint16([]byte{0x01, 0x02}) == binary.LittleEndian.Uint16([]byte{0x01, 0x02})

// tensorFloat32 decodes a tensor into float32 values, converting from the
// dtype the model was published with. POTION models store embeddings as F32
// except potion-code-16M-v2, which uses F16. When the F32 bytes can be used
// in place (little-endian host, 4-byte-aligned data), the returned slice
// aliases the tensor's backing buffer instead of copying it and aliased is
// true — with an mmapped file this is what lets concurrent processes share
// one physical copy of the weights.
func tensorFloat32(t safetensors.TensorView) (out []float32, aliased bool, err error) {
	data := t.Data()
	switch t.DType() {
	case safetensors.F32:
		if len(data) >= 4 && hostLittleEndian && uintptr(unsafe.Pointer(&data[0]))%4 == 0 {
			return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/4), true, nil
		}
		out = make([]float32, len(data)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return out, false, nil
	case safetensors.F16:
		out = make([]float32, len(data)/2)
		for i := range out {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(data[i*2:]))
		}
		return out, false, nil
	default:
		return nil, false, fmt.Errorf("unsupported tensor dtype: %s", t.DType())
	}
}

// loadEmbeddings loads the model weights from a safetensors file. Alongside
// the embedding matrix it returns the optional per-token weights (F64) and
// token-to-row mapping (I64) tensors used by vocabulary-quantized models.
//
// The file is memory-mapped rather than read into the heap, and on the
// common path (F32 weights, little-endian host) the returned embeddings
// slice aliases the mapping directly, so concurrent processes loading the
// same model share one physical copy of the weights through the OS page
// cache. The returned unmap function releases the mapping; it is non-nil
// exactly when embeddings aliases mapped memory, and must not be called
// while embeddings is still in use. Where mapping is unavailable or the
// bytes need conversion, the weights live on the Go heap and unmap is nil.
func loadEmbeddings(safetensorsPath string) (embeddings []float32, dimensions int, weights []float64, mapping []int64, unmap func() error, err error) {
	data, unmapData, mmapErr := mmapFile(safetensorsPath)
	if mmapErr != nil {
		data, err = os.ReadFile(safetensorsPath)
		if err != nil {
			return nil, 0, nil, nil, nil, err
		}
	}
	// Release the mapping unless the embeddings slice ends up aliasing it
	aliased := false
	defer func() {
		if unmapData != nil && !aliased {
			unmapData()
		}
	}()

	st, err := safetensors.Deserialize(data)
	if err != nil {
		return nil, 0, nil, nil, nil, err
	}

	tensor, ok := st.Tensor("embeddings")
	if !ok {
		return nil, 0, nil, nil, nil, fmt.Errorf("embeddings tensor not found")
	}
	embeddings, aliased, err = tensorFloat32(tensor)
	if err != nil {
		return nil, 0, nil, nil, nil, fmt.Errorf("embeddings: %w", err)
	}
	dimensions = int(tensor.Shape()[1])

	if t, ok := st.Tensor("weights"); ok {
		if t.DType() != safetensors.F64 {
			aliased = false
			return nil, 0, nil, nil, nil, fmt.Errorf("weights: unsupported tensor dtype: %s", t.DType())
		}
		raw := t.Data()
		weights = make([]float64, len(raw)/8)
		for i := range weights {
			weights[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
		}
	}

	if t, ok := st.Tensor("mapping"); ok {
		if t.DType() != safetensors.I64 {
			aliased = false
			return nil, 0, nil, nil, nil, fmt.Errorf("mapping: unsupported tensor dtype: %s", t.DType())
		}
		raw := t.Data()
		mapping = make([]int64, len(raw)/8)
		for i := range mapping {
			mapping[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
		}
	}

	if aliased {
		unmap = unmapData
	}
	return embeddings, dimensions, weights, mapping, unmap, nil
}

// loadTokenizer parses tokenizer.json and constructs the tokenizer
// implementation matching the file's model type.
func loadTokenizer(tokenizerPath string) (tokenizer, error) {
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, err
	}

	var file tokenizerFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}

	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(file.Model, &probe); err != nil {
		return nil, fmt.Errorf("tokenizer model section: %w", err)
	}

	switch probe.Type {
	case "Unigram":
		return newUnigramTokenizer(file)
	case "WordPiece", "":
		// Older tokenizer.json files omit the model type; WordPiece is
		// the historical default for POTION models
		return newWordPieceTokenizer(file)
	default:
		return nil, fmt.Errorf("unsupported tokenizer model type: %s", probe.Type)
	}
}

// resolveCacheDir returns the directory model files are cached in: the
// GO_POTION_HOME environment variable if set, otherwise the platform-native
// user cache, e.g. ~/Library/Caches/go-potion on macOS or ~/.cache/go-potion
// on Linux.
func resolveCacheDir() (string, error) {
	if env := os.Getenv("GO_POTION_HOME"); env != "" {
		return env, nil
	}
	userCache, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("cannot determine user cache directory: %w", err)
	}
	return filepath.Join(userCache, "go-potion"), nil
}

// New creates a new Potion instance with the specified model. Model files
// are downloaded on first use into a per-user cache directory shared across
// projects (os.UserCacheDir()/go-potion); set the GO_POTION_HOME environment
// variable to cache them elsewhere. The context bounds the downloads —
// cancelling it aborts an in-flight transfer without leaving a partial file
// in the cache. Once the model is cached, New performs no network I/O.
//
// Where the platform allows it, the model weights are memory-mapped
// read-only rather than copied onto the Go heap, so loading is near-instant
// and concurrent processes using the same model share one physical copy of
// the weights through the OS page cache. See Close.
func New(ctx context.Context, modelKind Model) (*Potion, error) {
	repo, ok := modelRepos[modelKind]
	if !ok {
		return nil, fmt.Errorf("unknown model kind: %s", modelKind)
	}
	// resolve/main follows LFS pointers, which some tokenizer.json files
	// (potion-multilingual-128M) are stored as; for regular files it
	// serves the same bytes as raw/main
	safetensorsURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/model.safetensors", repo)
	tokenizerURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/tokenizer.json", repo)

	cacheDir, err := resolveCacheDir()
	if err != nil {
		return nil, err
	}

	// Create model directory
	modelDir := filepath.Join(cacheDir, string(modelKind))
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return nil, err
	}

	// Download any missing files; each is checked independently so a
	// partially populated cache heals itself
	safetensorsPath := filepath.Join(modelDir, "model.safetensors")
	tokenizerPath := filepath.Join(modelDir, "tokenizer.json")

	if err := ensureFile(ctx, safetensorsURL, safetensorsPath); err != nil {
		return nil, err
	}
	if err := ensureFile(ctx, tokenizerURL, tokenizerPath); err != nil {
		return nil, err
	}

	// Best-effort: an unaligned file (e.g. in a read-only cache) still
	// loads correctly, just with slower encoding, and a corrupt one fails
	// with a better error in loadEmbeddings below
	_ = alignSafetensors(safetensorsPath)

	embeddings, dimensions, weights, mapping, unmap, err := loadEmbeddings(safetensorsPath)
	if err != nil {
		return nil, err
	}

	tok, err := loadTokenizer(tokenizerPath)
	if err != nil {
		if unmap != nil {
			unmap()
		}
		return nil, err
	}

	return &Potion{
		embeddings: embeddings,
		dimensions: dimensions,
		weights:    weights,
		mapping:    mapping,
		tok:        tok,
		unmap:      unmap,
	}, nil
}

// Close releases the memory-mapped model weights. The Potion must not be
// used after Close. Calling it is optional — a model that lives for the
// life of the process can skip it, since the OS reclaims the mapping at
// exit — but programs that load and discard many models should Close them,
// because mapped weights are invisible to the Go garbage collector. Close
// is a no-op when the weights live on the Go heap.
func (p *Potion) Close() error {
	if p.unmap == nil {
		return nil
	}
	unmap := p.unmap
	p.unmap = nil
	p.embeddings = nil
	return unmap()
}

// Dimensions returns the width of the embeddings this model produces, e.g.
// 64 for BASE2M. Useful for sizing a vector index before encoding anything.
func (p *Potion) Dimensions() int {
	return p.dimensions
}

// Tokenize converts a sentence into the token IDs model2vec would produce:
// WordPiece models drop unknown tokens, the Unigram model keeps them.
func (p *Potion) Tokenize(sentence string) ([]int, error) {
	if sentence == "" {
		return make([]int, 0), nil
	}
	return p.tok.tokenize(sentence)
}

// Encode converts a single sentence into an embedding
func (p *Potion) Encode(sentence string) ([]float32, error) {
	if sentence == "" {
		return make([]float32, p.dimensions), nil
	}

	tokens, err := p.tok.tokenize(sentence)
	if err != nil {
		return nil, err
	}

	// All tokens may be unknown (and dropped), e.g. text in a script the
	// vocabulary doesn't cover; return a zero vector like model2vec does
	if len(tokens) == 0 {
		return make([]float32, p.dimensions), nil
	}

	if p.weights != nil {
		return p.poolWeighted(tokens), nil
	}
	return p.poolMean(tokens), nil
}

// poolMean averages the embedding rows of tokens and L2-normalizes the
// result. Accumulation runs in float32, matching what model2vec (numpy) and
// model2vec-rs do for non-quantized models, directly into the output buffer
// so Encode allocates nothing beyond its result. Each row is sliced to
// len(out) before the inner loop so the compiler can prove the indexing in
// bounds and drop the per-element checks — this loop dominates Encode's
// runtime.
func (p *Potion) poolMean(tokens []int) []float32 {
	out := make([]float32, p.dimensions)
	if p.dimensions >= simdMinDims {
		for _, token := range tokens {
			f32.AccumulateAdd(out, p.rowOf(token), 0)
		}
	} else {
		// Short rows don't amortize the SIMD call overhead; sum four
		// token rows per pass instead to cut the load/store traffic on
		// out, which bounds this loop.
		k := 0
		for ; k+3 < len(tokens); k += 4 {
			a := p.rowOf(tokens[k])[:len(out)]
			b := p.rowOf(tokens[k+1])[:len(out)]
			c := p.rowOf(tokens[k+2])[:len(out)]
			d := p.rowOf(tokens[k+3])[:len(out)]
			for i := range out {
				out[i] += (a[i] + b[i]) + (c[i] + d[i])
			}
		}
		for ; k < len(tokens); k++ {
			a := p.rowOf(tokens[k])[:len(out)]
			for i := range out {
				out[i] += a[i]
			}
		}
	}

	// Average the embeddings
	n := float32(len(tokens))
	for i := range out {
		out[i] /= n
	}

	// Normalize
	var sum float64
	for _, v := range out {
		sum += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sum))
	for i := range out {
		out[i] /= norm
	}
	return out
}

// simdMinDims is the embedding width from which poolMean accumulates rows
// with SIMD (github.com/tphakala/simd, NEON/AVX with a pure-Go fallback).
// Below it, per-row call overhead outweighs the vectorization win and a
// scalar four-rows-per-pass loop is faster; measured crossover on an Apple
// M1 Max sits between potion-base-2M (64 dims) and potion-base-4M (128).
const simdMinDims = 128

// rowOf returns the embedding row for a token, applying the row mapping of
// vocabulary-quantized models when present.
func (p *Potion) rowOf(token int) []float32 {
	if p.mapping != nil {
		token = int(p.mapping[token])
	}
	start := token * p.dimensions
	return p.embeddings[start : start+p.dimensions]
}

// poolWeighted mirrors model2vec's reference computation for
// vocabulary-quantized models (CODE16M): those carry float64 per-token
// weights, so accumulation runs in float64.
func (p *Potion) poolWeighted(tokens []int) []float32 {
	acc := make([]float64, p.dimensions)
	for _, token := range tokens {
		vec := p.rowOf(token)[:len(acc)]
		w := p.weights[token]
		for i, v := range vec {
			acc[i] += float64(v) * w
		}
	}

	// Average the embeddings
	for i := range acc {
		acc[i] /= float64(len(tokens))
	}

	// Normalize
	n := l2(acc)
	out := make([]float32, p.dimensions)
	for i := range acc {
		out[i] = float32(acc[i] / n)
	}
	return out
}

// EncodeMany converts multiple sentences into their respective embeddings,
// spreading the work over GOMAXPROCS goroutines. Encoding is CPU-bound, so
// more workers than that would only add scheduling overhead.
func (p *Potion) EncodeMany(sentences []string) ([][]float32, error) {
	results := make([][]float32, len(sentences))
	workers := min(runtime.GOMAXPROCS(0), len(sentences))

	var (
		wg       sync.WaitGroup
		next     atomic.Int64
		mu       sync.Mutex
		firstErr error
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(sentences) {
					return
				}
				embedding, err := p.Encode(sentences[i])
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("sentence %d: %w", i, err)
					}
					mu.Unlock()
					return
				}
				results[i] = embedding
			}
		}()
	}
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}
