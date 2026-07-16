package potion

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// unigramTokenizer implements the SentencePiece Unigram pipeline used by
// potion-multilingual-128M: a normalizer chain (Precompiled charsmap,
// Replace and Strip rules), a non-splitting Metaspace pre-tokenizer, and
// Viterbi decoding over the piece vocabulary. Unlike the WordPiece models,
// unknown runs map to the unknown token and are kept: the Python Unigram
// binding exposes no unk_token attribute, so model2vec does not filter
// unknown IDs for this tokenizer family.
type unigramTokenizer struct {
	normalizers []spNormalizer

	// Metaspace configuration
	replacement   string
	prependScheme string

	scores       []float64
	tokenToID    map[string]int
	minScore     float64
	unkID        int // -1 when the model defines none
	byteFallback bool
	maxTokenLen  int // longest vocabulary entry, in bytes
}

// spNormalizer is one step of a tokenizer.json normalizer chain.
type spNormalizer interface {
	normalize(string) string
}

type sequenceNormalizerConfig struct {
	Normalizers []json.RawMessage `json:"normalizers"`
}

type replaceNormalizer struct {
	literal string         // non-empty for String patterns
	regex   *regexp.Regexp // set for Regex patterns
	content string
}

func (r *replaceNormalizer) normalize(s string) string {
	if r.regex != nil {
		return r.regex.ReplaceAllString(s, r.content)
	}
	return strings.ReplaceAll(s, r.literal, r.content)
}

// goWhitespaceClass is Unicode's White_Space set spelled out for Go's RE2
// syntax: Go's \s only covers ASCII whitespace, while the Rust regex crate
// that HuggingFace uses treats \s as full Unicode White_Space.
const goWhitespaceClass = `[\t\n\x0B\f\r \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]`

type stripNormalizer struct {
	left  bool
	right bool
}

func (n *stripNormalizer) normalize(s string) string {
	if n.left {
		s = strings.TrimLeftFunc(s, unicode.IsSpace)
	}
	if n.right {
		s = strings.TrimRightFunc(s, unicode.IsSpace)
	}
	return s
}

// parseSPNormalizers recursively flattens a tokenizer.json normalizer
// section into a list of steps, supporting the types POTION's Unigram
// model uses. Unknown types are an error so a new model config fails
// loudly instead of silently mis-tokenizing.
func parseSPNormalizers(raw json.RawMessage) ([]spNormalizer, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, err
	}

	switch head.Type {
	case "Sequence":
		var cfg sequenceNormalizerConfig
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, err
		}
		var out []spNormalizer
		for _, child := range cfg.Normalizers {
			steps, err := parseSPNormalizers(child)
			if err != nil {
				return nil, err
			}
			out = append(out, steps...)
		}
		return out, nil

	case "Precompiled":
		var cfg struct {
			Charsmap string `json:"precompiled_charsmap"`
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, err
		}
		p, err := newPrecompiled(cfg.Charsmap)
		if err != nil {
			return nil, err
		}
		return []spNormalizer{p}, nil

	case "Replace":
		var cfg struct {
			Pattern struct {
				String *string `json:"String"`
				Regex  *string `json:"Regex"`
			} `json:"pattern"`
			Content string `json:"content"`
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, err
		}
		switch {
		case cfg.Pattern.String != nil:
			return []spNormalizer{&replaceNormalizer{literal: *cfg.Pattern.String, content: cfg.Content}}, nil
		case cfg.Pattern.Regex != nil:
			// The reference implementation uses Rust's regex crate,
			// where \s means Unicode White_Space; expand it for RE2
			pattern := strings.ReplaceAll(*cfg.Pattern.Regex, `\s`, goWhitespaceClass)
			re, err := regexp.Compile(pattern)
			if err != nil {
				return nil, fmt.Errorf("replace pattern %q: %w", *cfg.Pattern.Regex, err)
			}
			return []spNormalizer{&replaceNormalizer{regex: re, content: cfg.Content}}, nil
		default:
			return nil, fmt.Errorf("replace normalizer without String or Regex pattern")
		}

	case "Strip":
		var cfg struct {
			Left  bool `json:"strip_left"`
			Right bool `json:"strip_right"`
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, err
		}
		return []spNormalizer{&stripNormalizer{left: cfg.Left, right: cfg.Right}}, nil

	default:
		return nil, fmt.Errorf("unsupported normalizer type: %s", head.Type)
	}
}

// sentencePiece is one vocabulary entry of a Unigram model, serialized in
// tokenizer.json as a [token, log-probability] pair.
type sentencePiece struct {
	token string
	score float64
}

func (p *sentencePiece) UnmarshalJSON(data []byte) error {
	var arr [2]any
	if err := json.Unmarshal(data, &arr); err != nil {
		return err
	}
	token, ok := arr[0].(string)
	if !ok {
		return fmt.Errorf("unigram vocab entry: token is not a string")
	}
	score, ok := arr[1].(float64)
	if !ok {
		return fmt.Errorf("unigram vocab entry %q: score is not a number", token)
	}
	p.token, p.score = token, score
	return nil
}

func newUnigramTokenizer(file tokenizerFile) (*unigramTokenizer, error) {
	var model struct {
		UnkID        *int            `json:"unk_id"`
		Vocab        []sentencePiece `json:"vocab"`
		ByteFallback bool            `json:"byte_fallback"`
	}
	if err := json.Unmarshal(file.Model, &model); err != nil {
		return nil, fmt.Errorf("unigram model section: %w", err)
	}
	if len(model.Vocab) == 0 {
		return nil, fmt.Errorf("unigram model has an empty vocabulary")
	}

	normalizers, err := parseSPNormalizers(file.Normalizer)
	if err != nil {
		return nil, fmt.Errorf("normalizer section: %w", err)
	}

	var pre struct {
		Type           string `json:"type"`
		Replacement    string `json:"replacement"`
		PrependScheme  string `json:"prepend_scheme"`
		Split          *bool  `json:"split"`
		AddPrefixSpace *bool  `json:"add_prefix_space"`
	}
	if err := json.Unmarshal(file.PreTokenizer, &pre); err != nil {
		return nil, fmt.Errorf("pre_tokenizer section: %w", err)
	}
	if pre.Type != "Metaspace" {
		return nil, fmt.Errorf("unsupported pre-tokenizer for unigram model: %s", pre.Type)
	}
	// Legacy configs express prepend_scheme:never as add_prefix_space:false
	scheme := pre.PrependScheme
	if scheme == "" {
		scheme = "always"
	}
	if pre.AddPrefixSpace != nil && !*pre.AddPrefixSpace {
		scheme = "never"
	}
	if pre.Split != nil && *pre.Split {
		return nil, fmt.Errorf("metaspace with split=true is not supported")
	}

	t := &unigramTokenizer{
		normalizers:   normalizers,
		replacement:   pre.Replacement,
		prependScheme: scheme,
		scores:        make([]float64, len(model.Vocab)),
		tokenToID:     make(map[string]int, len(model.Vocab)),
		minScore:      model.Vocab[0].score,
		unkID:         -1,
		byteFallback:  model.ByteFallback,
	}
	if model.UnkID != nil {
		if *model.UnkID < 0 || *model.UnkID >= len(model.Vocab) {
			return nil, fmt.Errorf("unigram unk_id %d outside vocabulary", *model.UnkID)
		}
		t.unkID = *model.UnkID
	}
	for id, piece := range model.Vocab {
		t.scores[id] = piece.score
		t.tokenToID[piece.token] = id
		if piece.score < t.minScore {
			t.minScore = piece.score
		}
		if len(piece.token) > t.maxTokenLen {
			t.maxTokenLen = len(piece.token)
		}
	}
	return t, nil
}

// kUnkPenalty is SentencePiece's penalty for characters not covered by any
// vocabulary entry: they score minScore - kUnkPenalty in the lattice.
const kUnkPenalty = 10.0

// viterbi finds the best-scoring segmentation of sentence into vocabulary
// pieces. It is a direct port of HuggingFace's Unigram::encode_optimized
// (itself SentencePiece's optimized decoder), including its tie-breaking:
// on equal scores the earlier-inserted node wins, and consecutive unknown
// characters fuse into a single piece.
func (t *unigramTokenizer) viterbi(sentence string) ([]string, error) {
	size := len(sentence)
	unkScore := t.minScore - kUnkPenalty

	type bestPathNode struct {
		id       int
		score    float64
		startsAt int // -1 = unreached
	}
	nodes := make([]bestPathNode, size+1)
	for i := range nodes {
		nodes[i].startsAt = -1
	}

	for startsAt := 0; startsAt < size; {
		scoreTillHere := nodes[startsAt].score
		_, mblen := utf8.DecodeRuneInString(sentence[startsAt:])
		hasSingleNode := false

		// Enumerate vocabulary entries starting here, shortest first,
		// exactly like the reference trie's common-prefix search
		limit := min(size, startsAt+t.maxTokenLen)
		for end := startsAt + 1; end <= limit; end++ {
			id, ok := t.tokenToID[sentence[startsAt:end]]
			if !ok {
				continue
			}
			candidate := t.scores[id] + scoreTillHere
			target := &nodes[end]
			if target.startsAt == -1 || candidate > target.score {
				target.score = candidate
				target.startsAt = startsAt
				target.id = id
			}
			if !hasSingleNode && end-startsAt == mblen {
				hasSingleNode = true
			}
		}

		if !hasSingleNode {
			if t.unkID < 0 {
				return nil, fmt.Errorf("cannot tokenize %q: no vocabulary entry and no unknown token", sentence[startsAt:startsAt+mblen])
			}
			target := &nodes[startsAt+mblen]
			candidate := unkScore + scoreTillHere
			if target.startsAt == -1 || candidate > target.score {
				target.score = candidate
				target.startsAt = startsAt
				target.id = t.unkID
			}
		}
		startsAt += mblen
	}

	// Backtrack, fusing consecutive unknown pieces (fuse_unk is always on
	// for Unigram models loaded from tokenizer.json)
	var reversed []string
	var unkRun []string
	for endsAt := size; endsAt > 0; {
		node := &nodes[endsAt]
		piece := sentence[node.startsAt:endsAt]
		if node.id == t.unkID && t.unkID >= 0 {
			unkRun = append(unkRun, piece)
		} else {
			if len(unkRun) > 0 {
				reversed = append(reversed, joinReversed(unkRun))
				unkRun = unkRun[:0]
			}
			reversed = append(reversed, piece)
		}
		endsAt = node.startsAt
	}
	if len(unkRun) > 0 {
		reversed = append(reversed, joinReversed(unkRun))
	}

	pieces := make([]string, 0, len(reversed))
	for i := len(reversed) - 1; i >= 0; i-- {
		pieces = append(pieces, reversed[i])
	}
	return pieces, nil
}

// joinReversed concatenates parts collected during right-to-left backtracking
// into their left-to-right string.
func joinReversed(parts []string) string {
	var sb strings.Builder
	for i := len(parts) - 1; i >= 0; i-- {
		sb.WriteString(parts[i])
	}
	return sb.String()
}

func (t *unigramTokenizer) tokenize(sentence string) ([]int, error) {
	s := sentence
	for _, n := range t.normalizers {
		s = n.normalize(s)
	}
	// An empty normalized string produces an empty pre-tokenization
	// (HuggingFace filters empty splits), hence no tokens
	if s == "" {
		return []int{}, nil
	}

	// Metaspace with split=false: the whole normalized text becomes a
	// single piece with spaces replaced by the meta character
	s = strings.ReplaceAll(s, " ", t.replacement)
	if t.prependScheme != "never" && !strings.HasPrefix(s, t.replacement) {
		s = t.replacement + s
	}

	pieces, err := t.viterbi(s)
	if err != nil {
		return nil, err
	}

	ids := make([]int, 0, len(pieces))
	for _, piece := range pieces {
		if id, ok := t.tokenToID[piece]; ok {
			ids = append(ids, id)
			continue
		}
		// A fused unknown run (or any out-of-vocabulary piece) falls
		// back to byte tokens when the model has them, else to unk
		if t.byteFallback {
			byteIDs, ok := t.byteTokenIDs(piece)
			if ok {
				ids = append(ids, byteIDs...)
				continue
			}
		}
		if t.unkID < 0 {
			return nil, fmt.Errorf("cannot map piece %q to a token id", piece)
		}
		ids = append(ids, t.unkID)
	}
	return ids, nil
}

// byteTokenIDs maps piece to its <0xXX> byte-fallback tokens, reporting
// false if any byte token is missing from the vocabulary.
func (t *unigramTokenizer) byteTokenIDs(piece string) ([]int, bool) {
	ids := make([]int, 0, len(piece))
	for i := 0; i < len(piece); i++ {
		id, ok := t.tokenToID[fmt.Sprintf("<0x%02X>", piece[i])]
		if !ok {
			return nil, false
		}
		ids = append(ids, id)
	}
	return ids, true
}
