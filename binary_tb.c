#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../primitives/include/mempool.h"
#include "../primitives/include/tensor.h"
#include "binary_fn.c"

//assuming only read
void memPoolAccess (MemPoolRequest* req, MemPoolResponse* resp){
	if(req->request_type = 2){
		uint32_t mem_pointer = req->argument_1;
		// not needed to complete
		// as we can assume mempool has
		// transfered the req bytes in
		// cache. So directly read from
		// buffer pointer;

	}
}
Tensor createTensor (uint32_t ndim, uint32_t* dims, TensorDataType dt, uint16_t mempool){
	TensorDescriptor td;
	Tensor tensor;
	uint32_t mem_pool_size = 1;

	td.row_major_form = 1;
	td.number_of_dimensions = ndim;

	for (int i = 0; i < ndim; ++i)
	{
		td.dimensions[i] = dims[i];
		mem_pool_size = mem_pool_size * dims[i] * 4; //size_of(dt) map from enum
	}

	tensor.descriptor = td;

	uint32_t* mem_pointer;

	mem_pointer = (uint32_t*) malloc(mem_pool_size);

	//initialize tensor with some values
	for (int i = 0; i < mem_pool_size/4; ++i)
	{
		mem_pointer[i] = i+mempool;
	}

	tensor.mem_pool_identifier = mempool;
	tensor.mem_pool_buffer_pointer = mem_pointer;

	return tensor;
}

void main(){
	const uint32_t ndim  = 2;
	uint32_t dims[ndim];
	dims[1] = 5;
	dims[0] = 5;

	//define tensor A

	Tensor A,B;
	A = createTensor(ndim, dims, u32,0);

	//define tensor B
	B = createTensor(ndim, dims, u32,1);

	uint32_t* mem = B.mem_pool_buffer_pointer;
	//call function

	binaryOperatorOnTensor(&A,&B,'+');

	//check B
	uint32_t mem_pool_size=1;
	for (int i = 0; i < ndim; ++i)
	{
		mem_pool_size = mem_pool_size * dims[i]; //size_of(dt) map from enum
	}
    mem = A.mem_pool_buffer_pointer;
	for (int i = 0; i < mem_pool_size; ++i)
	{
		if(mem[i] == 1+2*i){
			printf("pass %d\n",i);
		}else{
			printf("Error at %d  value = %d\n",i,mem[i]);
			break;
		}
	}
}
