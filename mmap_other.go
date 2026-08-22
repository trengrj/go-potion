//go:build !unix

package potion

import "errors"

// mmapFile is unavailable on this platform; callers fall back to reading
// the file into the heap.
func mmapFile(path string) ([]byte, func() error, error) {
	return nil, nil, errors.ErrUnsupported
}
