#include "binary_fn.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

int main(){
 	
}
// contiguous storage in memory pool for all the data types - assumed
// a_element_size >= b_element_size is assumed
void binaryOperatorOnTensor(Tensor* a, Tensor* b, Operation op) {
	// in-place unary operator: performs a = f(a) where f is specified by op
	// supported op --> sine, exp, ReLU, square, absolute
	TensorDescriptor td_a = a->descriptor;
	TensorDescriptor td_b = b->descriptor;

	uint32_t a_num_dim = td_a.number_of_dimensions;
	uint32_t b_num_dim = td_b.number_of_dimensions;

	// Tensor dimensions match check
	if(a_num_dim!= b_num_dim){
		printf("Tensors incompatible: Number of dimensions don't match");
		return;
	}

	TensorDataType a_dt = a->descriptor.data_type;
	TensorDataType b_dt = b->descriptor.data_type;

	uint32_t a_element_size = (sizeofTensorDataInBytes(a_dt));
	uint32_t b_element_size = (sizeofTensorDataInBytes(b_dt));
		
	uint32_t num_elems = 1; // product of dims (# of elements in tensor)  
	for(uint32_t i=0; i<a_n_dim; i+=1) {
		num_elems *= td_a.dimensions[i];
	}

	int total_dwords_a = (ceil((num_elems*a_element_size)/8.0)); //number of dwords of the tensor (assuming positive)
	int num_iter = (ceil(total_dwords_a*1.0/CACHE_SIZE)); // 63/16 --> in the loop 16 16 16 15
	
	for(int k=0; k<num_iter;k=k+1){

		int num_in_cache = 1; // number of elements in both CACHEs
		int num_dwords_stored_a = CACHE_SIZE; // number of dwords to be stored in CACHE a
		
		if((k == num_iter-1) && (total_dwords_a % CACHE_SIZE != 0)){
			num_in_cache = num_elems % ((CACHE_SIZE*8)/a_element_size);
			num_dwords_stored_a = total_dwords_a % CACHE_SIZE;
		}
		else{
			num_in_cache = (CACHE_SIZE*8)/a_element_size;
			num_dwords_stored_a = CACHE_SIZE;
		}

		int num_dwords_stored_b = (ceil(num_dwords_stored_a*((b_element_size)/(a_element_size)))); // number of dwords to be stored in CACHE b
		/////////////////////////////////////////////////
		// FIRST STAGE of Pipeline : Fetching from Memory 
		/////////////////////////////////////////////////

		MemPoolRequest req_a;
		req_a.request_type = READ;
		req_a.request_tag = 1; // ? not much used in READ and WRITE
		req_a.arguments[0] = num_dwords_stored_a;  //number of dwords requested 
		req_a.arguments[1] = a->mem_pool_buffer_pointer+ k*CACHE_SIZE; // start address
		req_a.arguments[2] = 1; // stride = 1 as pointwise
		MemPoolResponse mpr_a;
		memPoolAccess(a->mem_pool_identifier, &req_a, &mpr_a); // as in 104 of test_mempool.c

		MemPoolRequest req_b;
		req_b.request_type = READ;
		req_b.request_tag = 1; // ? not much used in READ and WRITE
		req_b.arguments[0] = num_dwords_stored_b;  //number of dwords requested 
		req_b.arguments[1] = b->mem_pool_buffer_pointer + k*CACHE_SIZE*((b_element_size)/(a_element_size)); // start address
		req_b.arguments[2] = 1; // stride = 1 as pointwise
		MemPoolResponse mpr_b;
		memPoolAccess(b->mem_pool_identifier, &req_b, &mpr); // as in 104 of test_mempool.c

		uint64_t store_here_a[CACHE_SIZE]; // initialized an empty array with required size for storing from copyTensor
		void *array_a;
		array_a = store_here_a;

		uint64_t store_here_b[CACHE_SIZE]; // initialized an empty array with required size for storing from copyTensor
		void *array_b;
		array_b = store_here_b;

		for(int i=0; i<num_dwords_stored_a; i=i+1){   // may lead to segmented fault
			copyTensorEntry(&td_a, array_a, i, mpr_a.read_data, i);
		}

		for(int i=0; i<num_dwords_stored_b; i=i+1){   // may lead to segmented fault
			copyTensorEntry(&td_b, array_b, i, mpr_b.read_data, i);
		}


		/////////////////////////////////////
		// SECOND STAGE of Pipeline : Compute  
		/////////////////////////////////////

		switch(op){ 

			case ADD : // a = a + b
				for(int j=0; j<num_in_cache; j+=1) {
					switch(a_dt){
						case u8:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint8_t val_a = *(((uint8_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint8_t*)array_a) + j) = val_a + val_b;
									break;

								case i8:
									// does this make sense?
									break;
							}
							break;

						case u16:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint16_t val_a = *(((uint16_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint16_t*)array_a) + j) = (uint16_t) val_a + val_b;
									break;

								case i8:
									// does this make sense?
									break;

								case float16:
									// does this make sense?
									break;

								case u16:
									uint16_t val_a = *(((uint16_t*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((uint16_t*)array_a) + j) = val_a + val_b;
									break;

								case i16:
									// does this make sense?
									break;
							}
							break;

						case u32:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint32_t val_a = *(((uint32_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint32_t*)array_a) + j) = (uint32_t) val_a + val_b;
									break;

								case i8:
									// does this make sense?
									break;

								case float16:
									// does this make sense?
									break;

								case u16:
									uint32_t val_a = *(((uint32_t*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((uint32_t*)array_a) + j) = (uint32_t) val_a + val_b;
									break;

								case i16:
									// does this make sense?
									break;

								case float32:
									// does this make sense?
									break;

								case u32:
									uint32_t val_a = *(((uint32_t*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((uint32_t*)array_a) + j) = val_a + val_b;
									break;

								case i32:
									// does this make sense?
									break;
							}
							break;

						case u64:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = (uint64_t) val_a + val_b;
									break;

								case i8:
									// does this make sense?
									break;

								case float16:
									// does this make sense?
									break;

								case u16:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = (uint64_t) val_a + val_b;
									break;

								case i16:
									// does this make sense?
									break;

								case float32:
									// does this make sense?

								case u32:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = (uint64_t) val_a + val_b;
									break;

								case i32:
									// does this make sense?
									break;

								case float64:
									// does this make sense?
									break;

								case u64:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint64_t val_b = *(((uint64_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = val_a + val_b;
									break;

								case i64:
									// does this make sense?
									break;
							}
							break;
							
						case i8:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									// does this make sense?
									break;

								case i8:
									int8_t val_a = *(((int8_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int8_t*)array_a) + j) = val_a + val_b;
									break;
							}
							break;

						case i16:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									int16_t val_a = *(((int16_t*)array_a) + j);
									int16_t val_b = (int16_t) *(((uint8_t*)array_b) + j);
									*(((int16_t*)array_a) + j) = val_a + val_b;
									break;

								case i8:
									int16_t val_a = *(((int16_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int16_t*)array_a) + j) = (int16_t) val_a + val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									// does this make sense?
									break;

								case i16:
									int16_t val_a = *(((int16_t*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((int16_t*)array_a) + j) = val_a + val_b;
									break;
							}
							break;

						case i32:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int16_t val_b = (int16_t) *(((uint8_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a + val_b;
									break;

								case i8:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a + val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int32_t val_b = (int32_t) *(((uint16_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = val_a + val_b;
									break;

								case i16:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a + val_b;
									break;

								case float32:
									int32_t val_a = *(((int32_t*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a + val_b;
									break;									

								case u32:
									// does this make sense?
									break;

								case i32:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = val_a + val_b;
									break;
							}
							break;

						case i64:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int16_t val_b = (int16_t) *(((uint8_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a + val_b;
									break;

								case i8:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a + val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int32_t val_b = (int32_t) *(((uint16_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a + val_b;
									break;

								case i16:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a + val_b;
									break;

								case float32:
									int64_t val_a = *(((int64_t*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a + val_b;
									break;	

								case u32:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int64_t val_b = (int64_t) *(((uint32_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = val_a + val_b;
									break;

								case i32:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a + val_b;
									break;

								case float64:
									int64_t val_a = *(((int64_t*)array_a) + j);
									double val_b = *(((double*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a + val_b;
									break;	

								case u64:
									// does this make sense?
									break;

								case i64:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int64_t val_b = *(((int64_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = val_a + val_b;
									break;
							}
							break;

						case float8:
							// to be added 
							break;

						case float16:
							// to be added 
							break;

						case float32:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									float val_a = *(((float*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a + val_b;
									break;

								case i8:
									float val_a = *(((float*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a + val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									float val_a = *(((float*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a + val_b;
									break;

								case i16:
									float val_a = *(((float*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a + val_b;
									break;

								case float32:
									float val_a = *(((float*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((float*)array_a) + j) = val_a + val_b;
									break;									

								case u32:
									float val_a = *(((float*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a + val_b;
									break;

								case i32:
									float val_a = *(((float*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a + val_b;
									break;
							}
							break;

						case float64:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									double val_a = *(((double*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;

								case i8:
									double val_a = *(((double*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									double val_a = *(((double*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;

								case i16:
									double val_a = *(((double*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;

								case float32:
									double val_a = *(((double*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;	

								case u32:
									double val_a = *(((double*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;

								case i32:
									double val_a = *(((double*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;

								case float64:
									double val_a = *(((double*)array_a) + j);
									double val_b = *(((double*)array_b) + j);
									*(((double*)array_a) + j) = val_a + val_b;
									break;	

								case u64:
									double val_a = *(((double*)array_a) + j);
									uint64_t val_b = *(((uint64_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;

								case i64:
									double val_a = *(((double*)array_a) + j);
									int64_t val_b = *(((int64_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a + val_b;
									break;
							}
							break;
					}
				}
				break;

// ----------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------

			case MULT : // a = a * b
				for(int j=0; j<num_in_cache; j+=1) {
					switch(a_dt){
						case u8:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint8_t val_a = *(((uint8_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint8_t*)array_a) + j) = val_a * val_b;
									break;

								case i8:
									// does this make sense?
									break;
							}
							break;

						case u16:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint16_t val_a = *(((uint16_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint16_t*)array_a) + j) = (uint16_t) val_a * val_b;
									break;

								case i8:
									// does this make sense?
									break;

								case float16:
									// does this make sense?
									break;

								case u16:
									uint16_t val_a = *(((uint16_t*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((uint16_t*)array_a) + j) = val_a * val_b;
									break;

								case i16:
									// does this make sense?
									break;
							}
							break;

						case u32:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint32_t val_a = *(((uint32_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint32_t*)array_a) + j) = (uint32_t) val_a * val_b;
									break;

								case i8:
									// does this make sense?
									break;

								case float16:
									// does this make sense?
									break;

								case u16:
									uint32_t val_a = *(((uint32_t*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((uint32_t*)array_a) + j) = (uint32_t) val_a * val_b;
									break;

								case i16:
									// does this make sense?
									break;

								case float32:
									// does this make sense?
									break;

								case u32:
									uint32_t val_a = *(((uint32_t*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((uint32_t*)array_a) + j) = val_a * val_b;
									break;

								case i32:
									// does this make sense?
									break;
							}
							break;

						case u64:
							switch(b_dt){
								case float8:
									// does this make sense?
									break;

								case u8:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = (uint64_t) val_a * val_b;
									break;

								case i8:
									// does this make sense?
									break;

								case float16:
									// does this make sense?
									break;

								case u16:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = (uint64_t) val_a * val_b;
									break;

								case i16:
									// does this make sense?
									break;

								case float32:
									// does this make sense?

								case u32:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = (uint64_t) val_a * val_b;
									break;

								case i32:
									// does this make sense?
									break;

								case float64:
									// does this make sense?
									break;

								case u64:
									uint64_t val_a = *(((uint64_t*)array_a) + j);
									uint64_t val_b = *(((uint64_t*)array_b) + j);
									*(((uint64_t*)array_a) + j) = val_a * val_b;
									break;

								case i64:
									// does this make sense?
									break;
							}
							break;
							
						case i8:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									// does this make sense?
									break;

								case i8:
									int8_t val_a = *(((int8_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int8_t*)array_a) + j) = val_a * val_b;
									break;
							}
							break;

						case i16:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									int16_t val_a = *(((int16_t*)array_a) + j);
									int16_t val_b = (int16_t) *(((uint8_t*)array_b) + j);
									*(((int16_t*)array_a) + j) = val_a * val_b;
									break;

								case i8:
									int16_t val_a = *(((int16_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int16_t*)array_a) + j) = (int16_t) val_a * val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									// does this make sense?
									break;

								case i16:
									int16_t val_a = *(((int16_t*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((int16_t*)array_a) + j) = val_a * val_b;
									break;
							}
							break;

						case i32:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int16_t val_b = (int16_t) *(((uint8_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a * val_b;
									break;

								case i8:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a * val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int32_t val_b = (int32_t) *(((uint16_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = val_a * val_b;
									break;

								case i16:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a * val_b;
									break;

								case float32:
									int32_t val_a = *(((int32_t*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((int32_t*)array_a) + j) = (int32_t) val_a * val_b;
									break;									

								case u32:
									// does this make sense?
									break;

								case i32:
									int32_t val_a = *(((int32_t*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((int32_t*)array_a) + j) = val_a * val_b;
									break;
							}
							break;

						case i64:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int16_t val_b = (int16_t) *(((uint8_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a * val_b;
									break;

								case i8:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a * val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int32_t val_b = (int32_t) *(((uint16_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a * val_b;
									break;

								case i16:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a * val_b;
									break;

								case float32:
									int64_t val_a = *(((int64_t*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a * val_b;
									break;	

								case u32:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int64_t val_b = (int64_t) *(((uint32_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = val_a * val_b;
									break;

								case i32:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a * val_b;
									break;

								case float64:
									int64_t val_a = *(((int64_t*)array_a) + j);
									double val_b = *(((double*)array_b) + j);
									*(((int64_t*)array_a) + j) = (int64_t) val_a * val_b;
									break;	

								case u64:
									// does this make sense?
									break;

								case i64:
									int64_t val_a = *(((int64_t*)array_a) + j);
									int64_t val_b = *(((int64_t*)array_b) + j);
									*(((int64_t*)array_a) + j) = val_a * val_b;
									break;
							}
							break;

						case float8:
							// to be added 
							break;

						case float16:
							// to be added 
							break;

						case float32:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									float val_a = *(((float*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a * val_b;
									break;

								case i8:
									float val_a = *(((float*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a * val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									float val_a = *(((float*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a * val_b;
									break;

								case i16:
									float val_a = *(((float*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a * val_b;
									break;

								case float32:
									float val_a = *(((float*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((float*)array_a) + j) = val_a * val_b;
									break;									

								case u32:
									float val_a = *(((float*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a * val_b;
									break;

								case i32:
									float val_a = *(((float*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((float*)array_a) + j) = (float) val_a * val_b;
									break;
							}
							break;

						case float64:
							switch(b_dt){
								case float8:
									// to be added
									break;

								case u8:
									double val_a = *(((double*)array_a) + j);
									uint8_t val_b = *(((uint8_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;

								case i8:
									double val_a = *(((double*)array_a) + j);
									int8_t val_b = *(((int8_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;

								case float16:
									// to be added
									break;

								case u16:
									double val_a = *(((double*)array_a) + j);
									uint16_t val_b = *(((uint16_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;

								case i16:
									double val_a = *(((double*)array_a) + j);
									int16_t val_b = *(((int16_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;

								case float32:
									double val_a = *(((double*)array_a) + j);
									float val_b = *(((float*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;	

								case u32:
									double val_a = *(((double*)array_a) + j);
									uint32_t val_b = *(((uint32_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;

								case i32:
									double val_a = *(((double*)array_a) + j);
									int32_t val_b = *(((int32_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;

								case float64:
									double val_a = *(((double*)array_a) + j);
									double val_b = *(((double*)array_b) + j);
									*(((double*)array_a) + j) = val_a * val_b;
									break;	

								case u64:
									double val_a = *(((double*)array_a) + j);
									uint64_t val_b = *(((uint64_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;

								case i64:
									double val_a = *(((double*)array_a) + j);
									int64_t val_b = *(((int64_t*)array_b) + j);
									*(((double*)array_a) + j) = (double) val_a * val_b;
									break;
							}
							break;
					}
				}
				break;

// ----------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------
			
			case SINE : // a = sin(a) // support only for float32 and float64 as of now 
				for(int j=0; j<num_in_cache; j+=1) {
					switch(a_dt){
						case float8: ;
							// to be added 
							break;

						case float16: ;
							// to be added 
							break;

						case float32: ;
							float x_s32;
							x_s32 = *(((float*)array) + j);
							*(((float*)array) + j) = sin(x_s32);
							break;

						case float64: ;
							double x_s64;
							x_s64 = *(((double*)array) + j);
							*(((double*)array) + j) = sin(x_s64);
							break;
					}
				}
				break;

			case ABSOLUTE : // a = |a|
			// for uint types --> i/p = o/p so do nothing 
			// for int types --> check MSB for sign 
			// for float types --> check MSB for sign
				for(int j=0; j<num_in_cache; j+=1) {
					switch(a_dt){
						case u8: ;
						case u16: ;
						case u32: ;
						case u64: ; 
							break;
						case i8: ;
							int8_t MSB8a = *(((int8_t*)array) + j);
							MSB8a = ((MSB8a>>7)&1);
							if(MSB8a == 1){
								*(((int8_t*)array) + j) *= -1;
							}
							break;
						case i16: ;
							int16_t MSB16a = *(((int16_t*)array) + j);
							MSB16a = ((MSB16a>>15)&1);
							if(MSB16a == 1){
								*(((int16_t*)array) + j) *= -1;
							}
							break;
						case i32: ; 
							int32_t MSB32a = *(((int32_t*)array) + j);
							MSB32a = ((MSB32a>>31)&1);
							if(MSB32a == 1){
								*(((int32_t*)array) + j) *= -1;
							}
							break;
						case i64: ;
							int64_t MSB64a = *(((int64_t*)array) + j);
							MSB64a = ((MSB64a>>63)&1);
							if(MSB64a == 1){
								*(((int64_t*)array) + j) *= -1;
							}
							break;
						case float8: ;
							// to be added 
							break;
						case float16: ;
							// to be added 
							break;
						case float32: ; // if I use uint here then what happen
							uint32_t MSBf32 = *(((uint32_t*)array) + j);
							MSBf32 = ((MSBf32>>31)&1);
							if(MSBf32 == 1){
								*(((float*)array) + j) *= -1;
							}
							break;
						
						case float64: ;
							uint64_t MSBf64 = *(((uint64_t*)array) + j);
							MSBf64 = ((MSBf64>>63)&1);
							if(MSBf64 == 1){
								*(((double*)array) + j) *= -1;
							}
							break;
					}
				}
				break;

			default  :
				fprintf(stderr,"Error: unaryOperatorOnTensor: unknown Operation.\n"); // wanted to add which operation was accessed [more info during debug]
				return;
		}

		/////////////////////////////////////////
		// THIRD STAGE of Pipeline : Writing Back 
		/////////////////////////////////////////

		req_a.request_type = WRITE;
		//req_a.write_data = store_here; 
		for(int i=0; i<num_dwords_stored ;i=i+1){
			req_a.write_data[i] = store_here[i];
		}
		memPoolAccess(a->mem_pool_identifier,&req_a,&mpr);
	}
	return; 
}
