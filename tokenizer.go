package potion

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Tokenizer converts text into token IDs without loading any embeddings.
// It is safe for concurrent use.
type Tokenizer struct {
	tok         tokenizer
	template    []templateItem
	templateErr error
}

type templateItem struct {
	sequence bool
	id       int
}

var repoPattern = regexp.MustCompile(`^[\w-][\w.-]*/[\w-][\w.-]*$`)

// NewTokenizer loads the tokenizer of a HuggingFace repository such as
// "lightonai/mLateOn", downloading only tokenizer.json.
func NewTokenizer(ctx context.Context, repo string) (*Tokenizer, error) {
	if !repoPattern.MatchString(repo) {
		return nil, fmt.Errorf("invalid HuggingFace repository: %q", repo)
	}
	dirName := strings.ReplaceAll(repo, "/", "--")
	for model, modelRepo := range modelRepos {
		if modelRepo == repo {
			dirName = string(model)
		}
	}
	cacheDir, err := resolveCacheDir()
	if err != nil {
		return nil, err
	}
	modelDir := filepath.Join(cacheDir, dirName)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return nil, err
	}
	tokenizerPath := filepath.Join(modelDir, "tokenizer.json")
	tokenizerURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/tokenizer.json", repo)
	if err := ensureFile(ctx, tokenizerURL, tokenizerPath); err != nil {
		return nil, err
	}
	return loadTokenizer(tokenizerPath)
}

// NewTokenizerFromFile loads a HuggingFace tokenizer.json from disk.
func NewTokenizerFromFile(path string) (*Tokenizer, error) {
	return loadTokenizer(path)
}

// Tokenize returns token IDs without special tokens.
func (t *Tokenizer) Tokenize(text string) ([]int, error) {
	if text == "" {
		return make([]int, 0), nil
	}
	return t.tok.tokenize(text)
}

// Encode returns token IDs wrapped in the post-processor's special tokens.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t.templateErr != nil {
		return nil, t.templateErr
	}
	ids, err := t.Tokenize(text)
	if err != nil {
		return nil, err
	}
	if t.template == nil {
		return ids, nil
	}
	out := make([]int, 0, len(ids)+len(t.template))
	for _, item := range t.template {
		if item.sequence {
			out = append(out, ids...)
		} else {
			out = append(out, item.id)
		}
	}
	return out, nil
}

func parsePostProcessor(raw json.RawMessage) ([]templateItem, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("post_processor section: %w", err)
	}
	switch head.Type {
	case "TemplateProcessing":
		var cfg struct {
			Single []struct {
				SpecialToken *struct {
					ID string `json:"id"`
				} `json:"SpecialToken"`
				Sequence *struct {
					ID string `json:"id"`
				} `json:"Sequence"`
			} `json:"single"`
			SpecialTokens map[string]struct {
				IDs []int `json:"ids"`
			} `json:"special_tokens"`
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("post_processor section: %w", err)
		}
		var template []templateItem
		for _, item := range cfg.Single {
			switch {
			case item.Sequence != nil:
				if item.Sequence.ID != "A" {
					return nil, fmt.Errorf("post_processor template references sequence %q", item.Sequence.ID)
				}
				template = append(template, templateItem{sequence: true})
			case item.SpecialToken != nil:
				special, ok := cfg.SpecialTokens[item.SpecialToken.ID]
				if !ok {
					return nil, fmt.Errorf("post_processor template references unknown special token %q", item.SpecialToken.ID)
				}
				for _, id := range special.IDs {
					template = append(template, templateItem{id: id})
				}
			default:
				return nil, fmt.Errorf("post_processor template item is neither a sequence nor a special token")
			}
		}
		return template, nil
	case "BertProcessing":
		var cfg struct {
			Sep [2]any `json:"sep"`
			Cls [2]any `json:"cls"`
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("post_processor section: %w", err)
		}
		cls, ok1 := cfg.Cls[1].(float64)
		sep, ok2 := cfg.Sep[1].(float64)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("post_processor BertProcessing: malformed cls/sep")
		}
		return []templateItem{{id: int(cls)}, {sequence: true}, {id: int(sep)}}, nil
	default:
		return nil, fmt.Errorf("unsupported post_processor type: %s", head.Type)
	}
}
