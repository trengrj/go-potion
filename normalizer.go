package potion

import (
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// bertNormalizer implements text normalization for BERT-style tokenization
type bertNormalizer struct {
	// Whether to do the bert basic cleaning:
	//   1. Remove any control characters
	//   2. Replace all sorts of whitespace by the classic one ` `
	CleanText bool
	// Whether to put spaces around chinese characters so they get split
	HandleChineseChars bool
	// Whether to strip accents (NFD decomposition, then drop combining marks).
	// HuggingFace's BertNormalizer defaults this to the Lowercase setting when
	// strip_accents is null in tokenizer.json.
	StripAccents bool
	// Whether to lowercase the input
	Lowercase bool
}

func newBertNormalizer(cleanText, handleChineseChars, stripAccents, lowercase bool) *bertNormalizer {
	return &bertNormalizer{
		CleanText:          cleanText,
		HandleChineseChars: handleChineseChars,
		StripAccents:       stripAccents,
		Lowercase:          lowercase,
	}
}

func defaultBertNormalizer() *bertNormalizer {
	return &bertNormalizer{
		CleanText:          true,
		HandleChineseChars: true,
		StripAccents:       true,
		Lowercase:          true,
	}
}

func isWhitespace(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return true
	default:
		return unicode.IsSpace(r)
	}
}

// isControl checks whether a character is a control character
// These are technically control characters but we count them as whitespace
func isControl(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return false
	default:
		return unicode.IsControl(r) || unicode.Is(unicode.Cf, r) || unicode.Is(unicode.Co, r)
	}
}

func isChineseChar(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B920 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}

func (n *bertNormalizer) normalize(input string) string {
	if input == "" {
		return input
	}

	if isASCII(input) {
		return n.normalizeASCII(input)
	}

	return n.normalizeUnicode(input)
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

func isASCIISpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\v' || b == '\f' || b == '\r'
}

// normalizeASCII fuses cleaning, lowercasing and trimming into a single
// byte-level pass. Accent stripping (NFD is an identity transform below
// 0x80) and Chinese character handling are no-ops for ASCII input.
func (n *bertNormalizer) normalizeASCII(input string) string {
	buf := make([]byte, 0, len(input))
	for i := 0; i < len(input); i++ {
		b := input[i]
		if n.CleanText {
			switch {
			case b == '\t' || b == '\n' || b == '\r':
				b = ' '
			case b < 0x20 || b == 0x7f: // NUL and control characters
				continue
			}
		}
		if n.Lowercase && b >= 'A' && b <= 'Z' {
			b += 'a' - 'A'
		}
		buf = append(buf, b)
	}

	start, end := 0, len(buf)
	for start < end && isASCIISpace(buf[start]) {
		start++
	}
	for end > start && isASCIISpace(buf[end-1]) {
		end--
	}
	return string(buf[start:end])
}

func (n *bertNormalizer) normalizeUnicode(input string) string {
	// Fused clean + chinese-char pass; ranging over the string decodes
	// runes in place without an up-front []rune conversion
	runes := make([]rune, 0, len(input))
	for _, r := range input {
		if n.CleanText {
			if r == 0 || r == 0xfffd || isControl(r) {
				continue
			}
			if isWhitespace(r) {
				runes = append(runes, ' ')
				continue
			}
		}
		if n.HandleChineseChars && isChineseChar(r) {
			// Boundary spaces are removed by the final trim
			runes = append(runes, ' ', r, ' ')
			continue
		}
		runes = append(runes, r)
	}

	if n.StripAccents {
		// Decompose to NFD and drop nonspacing marks (Mn), matching
		// HuggingFace's strip_accents. The result is left decomposed,
		// as in the reference implementation. string(runes) copies, so
		// runes can be reused as the output buffer.
		decomposed := norm.NFD.String(string(runes))
		out := runes[:0]
		for _, r := range decomposed {
			if unicode.Is(unicode.Mn, r) {
				continue
			}
			if n.Lowercase {
				r = unicode.ToLower(r)
			}
			out = append(out, r)
		}
		runes = out
	} else if n.Lowercase {
		for i, r := range runes {
			runes[i] = unicode.ToLower(r)
		}
	}

	start, end := 0, len(runes)
	for start < end && unicode.IsSpace(runes[start]) {
		start++
	}
	for end > start && unicode.IsSpace(runes[end-1]) {
		end--
	}
	return string(runes[start:end])
}
