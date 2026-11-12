//go:build !purego

#include "textflag.h"

// FADD Vd.S4, Vn.S4, Vm.S4 — the Go assembler has no mnemonic for the
// vector float add.
#define FADD(m, n, d) WORD $(0x4E20D400 | (m)<<16 | (n)<<5 | (d))

// func sumRowsNEON(out *float32, emb *float32, tokens *int, n int, dims int)
//
// Adds the n embedding rows emb[tokens[i]*dims:][:dims] into out[:dims].
// dims must be a positive multiple of 16.
//
// Rows are summed eight at a time, so out is loaded and stored once per
// eight rows rather than once per row, and the eight row streams give the
// CPU independent cache misses to overlap. Within a group the rows are still
// added one after another in token order — the chains of adds of different
// columns overlap, so this costs no throughput — which keeps the result
// bit-identical to summing row by row. (Prefetching the next group's rows
// bought only 2-3% on Apple silicon, so it is left out.)
TEXT ·sumRowsNEON(SB), NOSPLIT, $0-40
	MOVD out+0(FP), R0
	MOVD emb+8(FP), R1
	MOVD tokens+16(FP), R2
	MOVD n+24(FP), R3
	MOVD dims+32(FP), R4
	LSL  $2, R4, R5            // row size in bytes

	CMP $8, R3
	BLT tail

loop8:
	LDP.P 16(R2), (R6, R7)
	LDP.P 16(R2), (R8, R9)
	LDP.P 16(R2), (R10, R11)
	LDP.P 16(R2), (R12, R13)
	MADD  R5, R1, R6, R6       // row address = emb + token * row size
	MADD  R5, R1, R7, R7
	MADD  R5, R1, R8, R8
	MADD  R5, R1, R9, R9
	MADD  R5, R1, R10, R10
	MADD  R5, R1, R11, R11
	MADD  R5, R1, R12, R12
	MADD  R5, R1, R13, R13
	MOVD  R0, R14              // out cursor
	MOVD  R4, R15              // floats left in the row

inner8:
	VLD1   (R14), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R6), [V4.S4, V5.S4, V6.S4, V7.S4]
	FADD(4, 0, 0)
	FADD(5, 1, 1)
	FADD(6, 2, 2)
	FADD(7, 3, 3)
	VLD1.P 64(R7), [V16.S4, V17.S4, V18.S4, V19.S4]
	FADD(16, 0, 0)
	FADD(17, 1, 1)
	FADD(18, 2, 2)
	FADD(19, 3, 3)
	VLD1.P 64(R8), [V20.S4, V21.S4, V22.S4, V23.S4]
	FADD(20, 0, 0)
	FADD(21, 1, 1)
	FADD(22, 2, 2)
	FADD(23, 3, 3)
	VLD1.P 64(R9), [V24.S4, V25.S4, V26.S4, V27.S4]
	FADD(24, 0, 0)
	FADD(25, 1, 1)
	FADD(26, 2, 2)
	FADD(27, 3, 3)
	VLD1.P 64(R10), [V4.S4, V5.S4, V6.S4, V7.S4]
	FADD(4, 0, 0)
	FADD(5, 1, 1)
	FADD(6, 2, 2)
	FADD(7, 3, 3)
	VLD1.P 64(R11), [V16.S4, V17.S4, V18.S4, V19.S4]
	FADD(16, 0, 0)
	FADD(17, 1, 1)
	FADD(18, 2, 2)
	FADD(19, 3, 3)
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	FADD(20, 0, 0)
	FADD(21, 1, 1)
	FADD(22, 2, 2)
	FADD(23, 3, 3)
	VLD1.P 64(R13), [V24.S4, V25.S4, V26.S4, V27.S4]
	FADD(24, 0, 0)
	FADD(25, 1, 1)
	FADD(26, 2, 2)
	FADD(27, 3, 3)
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R14)
	SUBS   $16, R15
	BNE    inner8

	SUB $8, R3
	CMP $8, R3
	BGE loop8

tail:
	CBZ R3, done

loop1:
	MOVD.P 8(R2), R6
	MADD   R5, R1, R6, R6
	MOVD   R0, R14
	MOVD   R4, R15

inner1:
	VLD1   (R14), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R6), [V4.S4, V5.S4, V6.S4, V7.S4]
	FADD(4, 0, 0)
	FADD(5, 1, 1)
	FADD(6, 2, 2)
	FADD(7, 3, 3)
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R14)
	SUBS   $16, R15
	BNE    inner1

	SUBS $1, R3
	BNE  loop1

done:
	RET
