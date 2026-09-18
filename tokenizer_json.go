package potion

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
)

type jsonScanner struct {
	data []byte
	pos  int
}

func (s *jsonScanner) errorf(format string, args ...any) error {
	return fmt.Errorf("tokenizer.json at byte %d: %s", s.pos, fmt.Sprintf(format, args...))
}

func (s *jsonScanner) skipSpace() {
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case ' ', '\t', '\n', '\r':
			s.pos++
		default:
			return
		}
	}
}

func (s *jsonScanner) peek() byte {
	s.skipSpace()
	if s.pos >= len(s.data) {
		return 0
	}
	return s.data[s.pos]
}

func (s *jsonScanner) expect(c byte) error {
	if s.peek() != c {
		return s.errorf("expected %q", c)
	}
	s.pos++
	return nil
}

func (s *jsonScanner) rawString() (raw []byte, escaped bool, err error) {
	if s.peek() != '"' {
		return nil, false, s.errorf("expected string")
	}
	start := s.pos
	for i := s.pos + 1; i < len(s.data); i++ {
		switch s.data[i] {
		case '\\':
			escaped = true
			i++
		case '"':
			s.pos = i + 1
			return s.data[start:s.pos], escaped, nil
		}
	}
	return nil, false, s.errorf("unterminated string")
}

func (s *jsonScanner) appendString(buf []byte) ([]byte, error) {
	raw, escaped, err := s.rawString()
	if err != nil {
		return buf, err
	}
	if !escaped {
		return append(buf, raw[1:len(raw)-1]...), nil
	}
	var str string
	if err := json.Unmarshal(raw, &str); err != nil {
		return buf, s.errorf("%v", err)
	}
	return append(buf, str...), nil
}

func (s *jsonScanner) readString() (string, error) {
	raw, escaped, err := s.rawString()
	if err != nil {
		return "", err
	}
	if !escaped {
		return string(raw[1 : len(raw)-1]), nil
	}
	var str string
	if err := json.Unmarshal(raw, &str); err != nil {
		return "", s.errorf("%v", err)
	}
	return str, nil
}

func (s *jsonScanner) readInt() (int64, error) {
	s.skipSpace()
	start := s.pos
	neg := false
	if s.pos < len(s.data) && s.data[s.pos] == '-' {
		neg = true
		s.pos++
	}
	var v int64
	for s.pos < len(s.data) && s.data[s.pos] >= '0' && s.data[s.pos] <= '9' {
		v = v*10 + int64(s.data[s.pos]-'0')
		s.pos++
	}
	if s.pos == start || (neg && s.pos == start+1) {
		return 0, s.errorf("expected integer")
	}
	if s.pos < len(s.data) {
		switch s.data[s.pos] {
		case '.', 'e', 'E':
			return 0, s.errorf("expected integer, found a fraction or exponent")
		}
	}
	if neg {
		v = -v
	}
	return v, nil
}

func (s *jsonScanner) skipValue() ([]byte, error) {
	s.skipSpace()
	start := s.pos
	depth := 0
	for {
		c := s.peek()
		switch c {
		case 0:
			return nil, s.errorf("unexpected end of input")
		case '"':
			if _, _, err := s.rawString(); err != nil {
				return nil, err
			}
		case '{', '[':
			depth++
			s.pos++
		case '}', ']':
			depth--
			s.pos++
			if depth < 0 {
				return nil, s.errorf("unexpected %q", c)
			}
		case ',', ':':
			if depth == 0 {
				return nil, s.errorf("unexpected %q", c)
			}
			s.pos++
		default:
			for s.pos < len(s.data) {
				switch s.data[s.pos] {
				case ',', '}', ']', ' ', '\t', '\n', '\r':
					goto done
				}
				s.pos++
			}
		done:
		}
		if depth == 0 {
			return s.data[start:s.pos], nil
		}
	}
}

func (s *jsonScanner) object(fn func(key string) error) error {
	if err := s.expect('{'); err != nil {
		return err
	}
	if s.peek() == '}' {
		s.pos++
		return nil
	}
	for {
		key, err := s.readString()
		if err != nil {
			return err
		}
		if err := s.expect(':'); err != nil {
			return err
		}
		if err := fn(key); err != nil {
			return err
		}
		switch s.peek() {
		case ',':
			s.pos++
		case '}':
			s.pos++
			return nil
		default:
			return s.errorf("expected ',' or '}'")
		}
	}
}

func (s *jsonScanner) array(fn func() error) error {
	if err := s.expect('['); err != nil {
		return err
	}
	if s.peek() == ']' {
		s.pos++
		return nil
	}
	for {
		if err := fn(); err != nil {
			return err
		}
		switch s.peek() {
		case ',':
			s.pos++
		case ']':
			s.pos++
			return nil
		default:
			return s.errorf("expected ',' or ']'")
		}
	}
}

func splitTokenizerFile(data []byte) (tokenizerFile, error) {
	var file tokenizerFile
	s := &jsonScanner{data: data}
	err := s.object(func(key string) error {
		raw, err := s.skipValue()
		if err != nil {
			return err
		}
		switch key {
		case "normalizer":
			file.Normalizer = raw
		case "pre_tokenizer":
			file.PreTokenizer = raw
		case "model":
			file.Model = raw
		case "added_tokens":
			file.AddedTokens = raw
		case "post_processor":
			file.PostProcessor = raw
		}
		return nil
	})
	if err != nil {
		return file, err
	}
	if s.peek() != 0 {
		return file, s.errorf("trailing data after document")
	}
	return file, nil
}

func modelSectionType(raw []byte) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", fmt.Errorf("missing model section")
	}
	var typ string
	s := &jsonScanner{data: raw}
	err := s.object(func(key string) error {
		if key != "type" {
			_, err := s.skipValue()
			return err
		}
		var err error
		typ, err = s.readString()
		return err
	})
	return typ, err
}

type bpeModelSection struct {
	Vocab   map[string]int32
	Options struct {
		UnkToken                *string  `json:"unk_token"`
		ContinuingSubwordPrefix *string  `json:"continuing_subword_prefix"`
		EndOfWordSuffix         *string  `json:"end_of_word_suffix"`
		FuseUnk                 bool     `json:"fuse_unk"`
		ByteFallback            bool     `json:"byte_fallback"`
		IgnoreMerges            bool     `json:"ignore_merges"`
		Dropout                 *float64 `json:"dropout"`
	}
	merges []byte // raw merges array, parsed once the vocabulary is known
}

func parseBPEModelSection(raw []byte) (*bpeModelSection, error) {
	m := &bpeModelSection{Vocab: make(map[string]int32, 1<<16)}
	s := &jsonScanner{data: raw}
	options := []byte{'{'}
	err := s.object(func(key string) error {
		switch key {
		case "vocab":
			return s.object(func(tok string) error {
				id, err := s.readInt()
				if err != nil {
					return err
				}
				m.Vocab[tok] = int32(id)
				return nil
			})
		case "merges":
			v, err := s.skipValue()
			m.merges = v
			return err
		default:
			v, err := s.skipValue()
			if err != nil {
				return err
			}
			if len(options) > 1 {
				options = append(options, ',')
			}
			options = append(options, '"')
			options = append(options, key...)
			options = append(options, '"', ':')
			options = append(options, v...)
			return nil
		}
	})
	if err != nil {
		return nil, fmt.Errorf("bpe model section: %w", err)
	}
	options = append(options, '}')
	if err := json.Unmarshal(options, &m.Options); err != nil {
		return nil, fmt.Errorf("bpe model section: %w", err)
	}
	return m, nil
}

type mergeRule struct {
	a, b, ab int32
}

func (m *bpeModelSection) mergeRules() ([]mergeRule, error) {
	if m.merges == nil || string(m.merges) == "null" {
		return nil, nil
	}

	workers := max(1, min(runtime.GOMAXPROCS(0), 16))
	var starts []int // byte offset of each rule
	s := &jsonScanner{data: m.merges}
	err := s.array(func() error {
		s.skipSpace()
		starts = append(starts, s.pos)
		_, err := s.skipValue()
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("bpe merges: %w", err)
	}
	if len(starts) == 0 {
		return nil, nil
	}
	end := s.pos - 1 // the closing bracket
	starts = append(starts, end)

	rules := make([]mergeRule, len(starts)-1)
	errs := make([]error, workers)
	per := (len(rules) + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*per, min((w+1)*per, len(rules))
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[w] = m.resolveMerges(rules, lo, hi, starts)
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("bpe merges: %w", err)
		}
	}
	return rules, nil
}

func (m *bpeModelSection) resolveMerges(rules []mergeRule, lo, hi int, starts []int) error {
	var buf []byte
	for i := lo; i < hi; i++ {
		s := &jsonScanner{data: m.merges[:starts[i+1]], pos: starts[i]}
		var err error
		var split int
		buf = buf[:0]
		if s.peek() == '[' {
			n := 0
			err = s.array(func() error {
				if n == 1 {
					split = len(buf)
				}
				n++
				buf, err = s.appendString(buf)
				return err
			})
			if err == nil && n != 2 {
				err = s.errorf("merge %d is not a pair", i)
			}
		} else {
			buf, err = s.appendString(buf)
			if err == nil {
				split = -1
				for j, c := range buf {
					if c == ' ' {
						split = j
						break
					}
				}
				if split < 0 {
					return s.errorf("merge %d is not a pair", i)
				}
				buf = append(buf[:split], buf[split+1:]...)
			}
		}
		if err != nil {
			return err
		}
		a, ok := m.Vocab[string(buf[:split])]
		if !ok {
			return fmt.Errorf("merge %d: %q is not in the vocabulary", i, buf[:split])
		}
		b, ok := m.Vocab[string(buf[split:])]
		if !ok {
			return fmt.Errorf("merge %d: %q is not in the vocabulary", i, buf[split:])
		}
		ab, ok := m.Vocab[string(buf)]
		if !ok {
			return fmt.Errorf("merge %d: %q is not in the vocabulary", i, buf)
		}
		rules[i] = mergeRule{a: a, b: b, ab: ab}
	}
	return nil
}
