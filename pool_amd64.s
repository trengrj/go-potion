//go:build !purego

#include "textflag.h"

// func sumRowsAVX2(out *float32, emb *float32, tokens *int, n int, dims int)
//
// Adds the n embedding rows emb[tokens[i]*dims:][:dims] into out[:dims].
// dims must be a positive multiple of 16.
//
// Rows are summed four at a time, so out is loaded and stored once per four
// rows rather than once per row, and the four row streams give the CPU
// independent cache misses to overlap. Within a group the rows are still
// added one after another in token order, which keeps the result
// bit-identical to summing row by row.
TEXT ·sumRowsAVX2(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), DI
	MOVQ emb+8(FP), SI
	MOVQ tokens+16(FP), DX
	MOVQ n+24(FP), CX
	MOVQ dims+32(FP), BX
	SHLQ $2, BX                // row size in bytes

	CMPQ CX, $4
	JLT  tail

loop4:
	MOVQ  (DX), R8
	MOVQ  8(DX), R9
	MOVQ  16(DX), R10
	MOVQ  24(DX), R11
	ADDQ  $32, DX
	IMULQ BX, R8               // row address = emb + token * row size
	IMULQ BX, R9
	IMULQ BX, R10
	IMULQ BX, R11
	ADDQ  SI, R8
	ADDQ  SI, R9
	ADDQ  SI, R10
	ADDQ  SI, R11
	XORQ  AX, AX               // byte offset into the rows and out

inner4:
	VMOVUPS (DI)(AX*1), Y0
	VMOVUPS 32(DI)(AX*1), Y1
	VADDPS  (R8)(AX*1), Y0, Y0
	VADDPS  32(R8)(AX*1), Y1, Y1
	VADDPS  (R9)(AX*1), Y0, Y0
	VADDPS  32(R9)(AX*1), Y1, Y1
	VADDPS  (R10)(AX*1), Y0, Y0
	VADDPS  32(R10)(AX*1), Y1, Y1
	VADDPS  (R11)(AX*1), Y0, Y0
	VADDPS  32(R11)(AX*1), Y1, Y1
	VMOVUPS Y0, (DI)(AX*1)
	VMOVUPS Y1, 32(DI)(AX*1)
	ADDQ    $64, AX
	CMPQ    AX, BX
	JLT     inner4

	SUBQ $4, CX
	CMPQ CX, $4
	JGE  loop4

tail:
	TESTQ CX, CX
	JZ    done

loop1:
	MOVQ  (DX), R8
	ADDQ  $8, DX
	IMULQ BX, R8
	ADDQ  SI, R8
	XORQ  AX, AX

inner1:
	VMOVUPS (DI)(AX*1), Y0
	VMOVUPS 32(DI)(AX*1), Y1
	VADDPS  (R8)(AX*1), Y0, Y0
	VADDPS  32(R8)(AX*1), Y1, Y1
	VMOVUPS Y0, (DI)(AX*1)
	VMOVUPS Y1, 32(DI)(AX*1)
	ADDQ    $64, AX
	CMPQ    AX, BX
	JLT     inner1

	DECQ CX
	JNZ  loop1

done:
	VZEROUPPER
	RET
