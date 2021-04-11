#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "mempool.h"
#include "tensor.h"
#include "unary_fn.c"
	
const float pi = 3.14159265358979323846;

float absol (float x) {
	if(x>=0) {return x;}
	else {return -1.0*x;}
}

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
	uint32_t mem_pool_size = 4; // 4 bytes for uint32_t/float32 dt

	td.row_major_form = 1; // use row-major format for tensor storage
	td.number_of_dimensions = ndim;

	for (int i = 0; i < ndim; ++i)
	{
		td.dimensions[i] = dims[i];
		mem_pool_size = mem_pool_size * dims[i]; //size_of(dt) map from enum
	}

	tensor.descriptor = td;

	float* mem_pointer;

	mem_pointer = (float*) malloc(mem_pool_size);

	//initialize tensor with some values
	for (int i = 0; i < mem_pool_size/4; ++i)
	{
		mem_pointer[i] = i*1.0*pi/16;
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

	Tensor A;
	A = createTensor(ndim, dims, float32, 0);

	uint32_t* mem = A.mem_pool_buffer_pointer;
	//call function

	unaryOperatorOnTensor(&A,'s');

	uint32_t mem_pool_size = 1; // # of elems in tensor

	for (int i = 0; i < ndim; ++i)
	{
		mem_pool_size = mem_pool_size * dims[i]; //size_of(dt) map from enum
	}

    for (int i = 0; i < mem_pool_size; ++i)
	{
		if(absol((float)(sin(i*1.0*pi/16)-mem[i])) <= 0.01){
			printf("pass %d\n",i);
		}
		else{
			printf("Error at %d  value = %d\n",i,mem[i]);
			break;
		}
	}
}
