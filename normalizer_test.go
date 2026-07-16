package potion

import (
	"testing"
)

func TestDefaultBertNormalizer(t *testing.T) {
	normalizer := defaultBertNormalizer()
	if !normalizer.CleanText {
		t.Error("Default CleanText should be true")
	}
	if !normalizer.HandleChineseChars {
		t.Error("Default HandleChineseChars should be true")
	}
	if !normalizer.Lowercase {
		t.Error("Default Lowercase should be true")
	}
}

func TestNewBertNormalizer(t *testing.T) {
	normalizer := newBertNormalizer(false, false, false, false)
	if normalizer.CleanText {
		t.Error("CleanText should be false")
	}
	if normalizer.HandleChineseChars {
		t.Error("HandleChineseChars should be false")
	}
	if normalizer.StripAccents {
		t.Error("StripAccents should be false")
	}
	if normalizer.Lowercase {
		t.Error("Lowercase should be false")
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
		config   *bertNormalizer
	}{
		{
			name:     "empty string",
			input:    "",
			expected: "",
			config:   defaultBertNormalizer(),
		},
		{
			name:     "basic cleaning",
			input:    "Hello\tWorld",
			expected: "hello world",
			config:   defaultBertNormalizer(),
		},
		{
			name:     "chinese characters",
			input:    "Hello世界",
			expected: "hello 世  界",
			config:   defaultBertNormalizer(),
		},
		{
			name:     "lowercase",
			input:    "HELLO",
			expected: "hello",
			config:   defaultBertNormalizer(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.normalize(tt.input)
			if result != tt.expected {
				t.Errorf("Normalize(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestIsWhitespace(t *testing.T) {
	tests := []struct {
		r        rune
		expected bool
	}{
		{' ', true},
		{'\t', true},
		{'\n', true},
		{'\r', true},
		{'a', false},
		{'世', false},
		{'\u2003', true}, // em space
	}

	for _, tt := range tests {
		t.Run(string(tt.r), func(t *testing.T) {
			result := isWhitespace(tt.r)
			if result != tt.expected {
				t.Errorf("isWhitespace(%q) = %v, want %v", tt.r, result, tt.expected)
			}
		})
	}
}

func TestIsControl(t *testing.T) {
	tests := []struct {
		r        rune
		expected bool
	}{
		{'\x00', true},
		{'\x1F', true},
		{'\t', false},
		{'\n', false},
		{'\r', false},
		{'a', false},
		{'世', false},
	}

	for _, tt := range tests {
		t.Run(string(tt.r), func(t *testing.T) {
			result := isControl(tt.r)
			if result != tt.expected {
				t.Errorf("isControl(%q) = %v, want %v", tt.r, result, tt.expected)
			}
		})
	}
}

func TestIsChineseChar(t *testing.T) {
	tests := []struct {
		r        rune
		expected bool
	}{
		{'世', true}, // CJK Unified Ideographs
		{'界', true}, // CJK Unified Ideographs
		{'a', false},
		{'あ', false}, // Hiragana
		{'ア', false}, // Katakana
		{'가', false}, // Hangul
	}

	for _, tt := range tests {
		t.Run(string(tt.r), func(t *testing.T) {
			result := isChineseChar(tt.r)
			if result != tt.expected {
				t.Errorf("isChineseChar(%q) = %v, want %v", tt.r, result, tt.expected)
			}
		})
	}
}
