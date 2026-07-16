// Package potion computes static text embeddings from the POTION model
// family (https://huggingface.co/collections/minishlab/potion), producing
// the same vectors as the Python model2vec library.
//
// A static embedding is the L2-normalized mean of per-token vectors: text
// is tokenized (WordPiece for most models, SentencePiece Unigram for the
// multilingual one), each token ID looked up in a pre-computed embedding
// matrix, and the rows averaged. No neural network runs at inference time,
// which makes encoding orders of magnitude faster than a transformer at
// the cost of context-insensitive embeddings.
//
//	encoder, err := potion.New(ctx, potion.BASE2M)
//	embedding, err := encoder.Encode("hello world")
//
// Model files are downloaded from HuggingFace on first use and cached under
// os.UserCacheDir()/go-potion (override with the GO_POTION_HOME environment
// variable), so New needs network access only once per machine.
//
// A Potion is immutable after New and safe for concurrent use; Encode may
// be called from any number of goroutines. Embeddings are L2-normalized, so
// the dot product of two embeddings is their cosine similarity.
package potion
