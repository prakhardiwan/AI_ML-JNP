#include "../../../../C/mempool/include/mempool.h"
#include "../../../../C/primitives/include/tensor.h"
#include "../../../../C/primitives/src/tensor.c"
#include "../../../../C/mempool/src/mempool.c"

// as of now float8 and float16 are not added
// SINE and EXP are calculated upto 9 terms in the expansion 
// 
typedef enum {
	SINE, 
	EXP,
	RELU,
	SQUARE,
    ABSOLUTE
} Operation;
