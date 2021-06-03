#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "mempool.c"
// #include "../../../C/mempool/src/mempool.c" // uncomment when pushing
// #include "tensor.h"
#include "unary_fn.c"
#include "createTensor.c"
// #include "../createTensor/src/createTensor.c" // uncomment when pushing

#define NPAGES 8

MemPool 	pool;
MemPoolRequest 	req;
MemPoolResponse resp;

Tensor *a;
// Tensor *b;

int _err_ = 0;

void main(){

    initMemPool(&pool,1,NPAGES);

	//define tensor
	const TensorDataType dataType = float32;
	const int8_t row_major_form = 1;
	const uint32_t ndim  = 2;
	
	uint32_t dims[ndim];
	dims[0] = 5;
	dims[1] = 5;

	const Operation operation = RELU;
	TensorDescriptor td_a;

	td_a.data_type = dataType;
	td_a.row_major_form = row_major_form;
	td_a.number_of_dimensions = ndim;

	// td_b.data_type = dataType;
	// td_b.row_major_form = row_major_form;
	// td_b.number_of_dimensions = ndim;

	for (int i = 0; i < ndim; i++)
	{
		td_a.dimensions[i] = dims[i];
		// td_b.dimensions[i] = dims[i];
	}

	//create tensor
    _err_ = createTensor(a,&pool,&req,&resp) + _err_;
    // _err_ = createTensor(b,&pool,&req,&resp) || _err_;

    if(_err_!=0)
		fprintf(stderr,"create Tensor FAILURE.\n");


	uint32_t element_size = sizeofTensorDataInBytes(td_a); 

	uint32_t num_elems = 1; // product of dims (# of elements in tensor)  

	for(uint32_t i=0; i<ndim; i++) {
		num_elems *= td_a.dimensions[i];
	}
	
	//fill tensor A values
	doublt offset = -20;
	fillTensorValues(a, num_elems, offset);

	// fillTensorValues(b, num_elems, 1);

	//call the function
	unaryOperatorOnTensor(a, operation);

	//check A (results)
	req.request_type = READ;

	int iter = 0;
	int elements_left = ceil((num_elems*element_size)/8.0);

	for( ; elements_left > 0; elements_left -= MAX_SIZE_OF_REQUEST_IN_WORDS){
		int elementsToRead = min(elements_left,MAX_SIZE_OF_REQUEST_IN_WORDS);
		req.request_tag = 1; // confirm dis
		req.arguments[0] =  elementsToRead; 
		req.arguments[1] = a->mem_pool_buffer_pointer+MAX_SIZE_OF_REQUEST_IN_WORDS*iter;
		req.arguments[2] = 1; // stride = 1 as pointwise
		iter ++;
		
		memPoolAccess(a->mem_pool_identifier,&req,&resp); 
		
		if(resp.status == NOT_OK) {
			fprintf(stderr,"read Tensor FAILURE.\n");
			return 0;
		}

		void *array1;
		array1 = resp.read_data;  

		for (int i = 0; i < elementsToRead; i++)
		{
		double operand1;

		//change expected 
		double expected_result = i+ offset + iter*MAX_SIZE_OF_REQUEST_IN_WORDS;   //or load from file
		switch (operation){
			case RELU: expected_result = (expected_result>0)? expected_result:0; break;
			case SINE: expected_result = sin(expected_result); break;
			case SQUARE: expected_result *= expected_result;
			case ABSOLUTE: expected_result = abs(expected_result);
		}

		switch(dataType){
			// case u8: ; 
			// 	uint8_t resultu8,ex_resultu8;
			// 	resultu8 = (uint8_t) *(((uint8_t*)array1) + i);
			// 	ex_resultu8 = (uint8_t) expected_result;
			// 	if (resultu8 != ex_resultu8){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;

			// case u16: ;
			// 	uint16_t resultu16,ex_resultu16;
			// 	resultu16 = (uint16_t) *(((uint16_t*)array1) + i);
			// 	ex_resultu16 = (uint16_t) expected_result;
			// 	if (resultu16 != ex_resultu16){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;

			// case u32: ;
			// 	uint32_t resultu32,ex_resultu32;
			// 	resultu32 = (uint32_t) *(((uint32_t*)array1) + i);
			// 	ex_resultu32 = (uint32_t) expected_result;
			// 	if (resultu32 != ex_resultu32){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;

			// case u64: ; 
			// 	uint64_t resultu64,ex_resultu64;
			// 	resultu64 = (uint64_t) *(((uint64_t*)array1) + i);
			// 	ex_resultu64 = (uint64_t) expected_result;
			// 	if (resultu64 != ex_resultu64){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;
				
			// case i8: ;
			// 	int8_t resulti8,ex_resulti8;
			// 	resulti8 = (int8_t) *(((int8_t*)array1) + i);
			// 	ex_resulti8 = (int8_t) expected_result;
			// 	if (resulti8 != ex_resulti8){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;

			// case i16: ;
			// 	int16_t resulti16,ex_resulti16;
			// 	resulti16 = (int16_t) *(((int16_t*)array1) + i);
			// 	ex_resulti16 = (int16_t) expected_result;
			// 	if (resulti16 != ex_resulti16){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;

			// case i32: ; 
			// 	int32_t resulti32,ex_resulti32;
			// 	resulti32 = (int32_t) *(((int32_t*)array1) + i);
			// 	ex_resulti32 = (int32_t) expected_result;
			// 	if (resulti32 != ex_resulti32){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;

			// case i64: ;
			// 	int64_t resulti64,ex_resulti64;
			// 	resulti64 = (int64_t) *(((int64_t*)array1) + i);
			// 	ex_resulti64 = (int64_t) expected_result;
			// 	if (resulti64 != ex_resulti64){
			// 		printf("fail at %d, iter = %d\n",i,iter);
			// 	}
			// 	break;

			// case float8: ;
				// to be added 
				// break;

			// case float16: ;
				// to be added 
				// break;

			case float32: ;
				float resultf32,ex_resultf32;
				resultf32 = (float) *(((float*)array1) + i);
				ex_resultf32 = (float) expected_result;
				if (resultf32 != ex_resultf32){ // check abs(resultf32-ex_resultf32)<eps
					printf("fail at %d, iter = %d\n Diff = %.10f\n",i,iter, resultf32-ex_resultf32);
				}
				break;

			case float64: ;
				double resultf64,ex_resultf64;
				resultf64 = (double) *(((double*)array1) + i);
				ex_resultf64 = (double) expected_result;
				if (resultf64 != ex_resultf64){
					printf("fail at %d, iter = %d\n Diff = %.10f\n",i,iter, resultf32-ex_resultf32);
				}
				break;			
		}
	}	
}
}


int fillTensorValues (Tensor* t,uint32_t num_elems, double offset ){
	uint32_t element_size = sizeofTensorDataInBytes(t->descriptor.data_type); 
	TensorDataType dataType = t->descriptor.data_type;

	req.request_type = WRITE;
	req.request_tag = 1; // confirm dis

	int iter = 0;
	int elements_left = ceil((num_elems*element_size)/8.0);
	for( ; elements_left > 0; elements_left -= MAX_SIZE_OF_REQUEST_IN_WORDS){
		int elementsToWrite = min(elements_left,MAX_SIZE_OF_REQUEST_IN_WORDS);
		req.arguments[0] =  elementsToWrite; 
		req.arguments[1] = t->mem_pool_buffer_pointer+MAX_SIZE_OF_REQUEST_IN_WORDS*iter;
		req.arguments[2] = 1; // stride = 1 as pointwise
		iter ++;

		void *array;
		array = req.write_data;  

		for (int i = 0; i < elementsToWrite; i++)
		{
		double data;
		data =  i + offset + iter * MAX_SIZE_OF_REQUEST_IN_WORDS; //or read from FILE
			switch(dataType){
			case u8: ; 
				uint8_t val8 = (uint8_t) data;
				*(((uint8_t*)array) + i) = val8;
				break;

			case u16: ;
				uint16_t val16 = (uint16_t) data;
				*(((uint16_t*)array) + i) = val16;
				break;

			case u32: ;
				uint32_t val32 = (uint32_t) data;
				*(((uint32_t*)array) + i) = val32;
				break;

			case u64: ; 
				uint64_t val64 = (uint64_t) data;
				*(((uint64_t*)array) + i) = val64;
				break;
				
			case i8: ;
				int8_t val8i = (int8_t) data;
				*(((int8_t*)array) + i) = val8i;
				break;

			case i16: ;
				int16_t val16i = (int16_t) data;
				*(((int16_t*)array) + i) = val16i;
				break;

			case i32: ; 
				int32_t val32i = (int32_t) data ;
				*(((int32_t*)array) + i) = val32i;
				break;

			case i64: ;
				int64_t val64i = (int64_t) data;
				*(((int64_t*)array) + i) = val64i;
				break;

			// case float8: ;
				// to be added 
				// break;

			// case float16: ;
				// to be added 
				// break;

			case float32: ;
				float val32f = (float) data;
				*(((float*)array) + i) = val32f;
				break;

			case float64: ;
				double val64f = (double) data;
				*(((double*)array) + i) = val64f;
				break;
				
			}		
		}

		memPoolAccess(t->mem_pool_identifier, &req, &resp); 

		if(resp.status = OK) {
			return 0;
		}else {
			fprintf(stderr,"write Tensor FAILURE.\n");
			return -1;
		}
	}	
}
