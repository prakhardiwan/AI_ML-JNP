#include "../../../../C/mempool/include/mempool.h"
#include "../../../../C/primitives/include/tensor.h"
#include "../../../../C/primitives/src/tensor.c"
#include "../../../../C/mempool/src/mempool.c"

/* Unary Operator:
unaryOperatorOnTensor(Tensor* a, Operation op): inplace pointwise unary operator for tensors
Operators: Sine, Exp, ReLU, Square, Absolute
The operator returns the required output (like sin(a), exp(a),... etc.) 
Datatypes supported for operators: 
SINE: float32, float64
EXP: float32, float64
ReLU: float8, float16, float32, float64, u8, u16, u32, u64, i8, i16, i32, i64
Square: float32, float64, u8, u16, u32, u64, i8, i16, i32, i64
Absolute: float32, float64, u8, u16, u32, u64, i8, i16, i32, i64

input: tensor, operation; output: modified tensor

3-staged pipeline: Fetch, Compute, Writeback

as of now float8 and float16 are not added
SINE and EXP functions used from the C math library 
*/ 

#define CACHE_SIZE      16 // in dwords so CACHE_SIZE*8 bytes 

typedef enum {
	SINE, 
	EXP,
	RELU,
	SQUARE,
    ABSOLUTE
} Operation;