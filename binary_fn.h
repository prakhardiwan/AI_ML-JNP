#include "../../../../C/mempool/include/mempool.h"
#include "../../../../C/primitives/include/tensor.h"
#include "../../../../C/primitives/src/tensor.c"
#include "../../../../C/mempool/src/mempool.c"

/* Unary Operator:
binaryOperatorOnTensor(Tensor* a, Tensor*b, Operation op): inplace pointwise binary operator for tensors

Assumption(s): size of datatype associated w/ Tensor a >= size of datatype associated w/ Tensor b

Operations: addition, multiplication, division

The function performs the required operation (for example, a = a + b, a = a * b, etc.) 

Illegal & illogical function calls: 

u8  = u8  + float8
u8  = u8  + int8

u16 = u16 + float8
u16 = u16 + int8
u16 = u16 + float16
u16 = u16 + int16

u32 = u32 + float8
u32 = u32 + int8
u32 = u32 + float16
u32 = u32 + int16
u32 = u32 + float32
u32 = u32 + int32

u64 = u64 + float8
u64 = u64 + int8
u64 = u64 + float16
u64 = u64 + int16
u64 = u64 + float32
u64 = u64 + int32
u64 = u64 + float64
u64 = u64 + int64

i8  = i8  + u8
i16 = i16 + u16
i32 = i32 + u32
i64 = i64 + u64

Input: 2 Tensors, operation
Output: A Tensor modified holding result of the operation

*/

// Support for float8 and float16 to be added

typedef enum {
	ADD, 
	MULT,
	DIV
} Operation;