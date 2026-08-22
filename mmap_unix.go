//go:build unix

package potion

import (
	"fmt"
	"os"
	"syscall"
)

// mmapFile maps the file at path into memory read-only and returns the
// mapped bytes with a function that releases the mapping. Because the
// mapping is backed by the OS page cache, every process that maps the same
// file shares one physical copy of it machine-wide, and no read of the file
// happens until the pages are first touched. Model files in the cache are
// only ever replaced by atomic rename (see downloadFile), so an existing
// mapping keeps its inode and can never observe a truncated file.
func mmapFile(path string) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	size := info.Size()
	if size <= 0 || int64(int(size)) != size {
		return nil, nil, fmt.Errorf("cannot mmap %s: invalid size %d", path, size)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
