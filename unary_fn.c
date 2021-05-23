#include "unary_fn.h"
#include<stdint.h>
#include <stdio.h>
#include <math.h>

float setrange_float (float x) {
	float pi = 3.14159265358979323846;
    int temp = (x/(2.0*pi)) + 0.5;
    x -= temp*2*pi;
    return x;
}

double setrange_double (double x) {
	double pi = 3.14159265358979323846;
    int temp = (x/(2.0*pi)) + 0.5;
    x -= temp*2*pi;
    return x;
}

// int main(){
// 	
// }
// contiguous storage in memory pool for all the data types - assumed
void unaryOperatorOnTensor(Tensor* a, Operation op) {
	// in-place unary operator: performs a = f(a) where f is specified by op
	// supported op --> sine, exp, ReLU, square, absolute

	TensorDescriptor td_a = a->descriptor;
	TensorDataType a_dt = a->descriptor.data_type;
	uint32_t element_size = sizeofTensorDataInBytes(a_dt); 
	uint32_t n_dim = td_a.number_of_dimensions; // number of dimensions

	uint32_t num_elems = 1; // product of dims (# of elements in tensor)  

	for(uint32_t i=0; i<n_dim; i+=1) {
		num_elems *= td_a.dimensions[i];
	}
	MemPoolRequest req_a;
	req_a.request_type = READ;
	req_a.request_tag = 1; // ? not much used in READ and WRITE
	req_a.arguments[0] = (ceil((num_elems*element_size)/8.0)); //number of dwords requested 
	req_a.arguments[1] = a->mem_pool_buffer_pointer;
	req_a.arguments[2] = 1; // stride = 1 as pointwise
	MemPoolResponse mpr;
	memPoolAccess(a->mem_pool_identifier,&req_a,&mpr); // as in 104 of test_mempool.c
	
	uint64_t store_here[req_a.arguments[0]]; // initialized an empty array with required size for storing from copyTensor. 
	void *array;
	array = store_here;  
	for(int i=0; i<num_elems; i=i+1){
		copyTensorEntry(&td_a,array,i,mpr.read_data,i);
	}
					
	switch(op){ 
		case RELU : // a = RELU(a)
		// for uint types --> i/p = o/p so do nothing 
		// for int types --> check MSB for sign 
		// for float types --> check MSB for sign 
			for(int j=0; j<num_elems; j+=1) {
				switch(a_dt){
					case u8: ;
					case u16: ;
					case u32: ;
					case u64: ; 
						break;
					case i8: ; 
					case float8: ;
						uint8_t MSB8 = *(((uint8_t*)array) + j);
						MSB8 = ((MSB8>>7)&1);
						if(MSB8 == 1){
							*(((uint8_t*)array) + j) = 0;
						}
						break;
					case i16: ;
					case float16: ;
						uint16_t MSB16 = *(((uint16_t*)array) + j);
						MSB16 = ((MSB16>>15)&1);
						if(MSB16 == 1){
							*(((uint16_t*)array) + j) = 0;
						}
						break;
					case i32: ; 
					case float32: ;
						uint32_t MSB32 = *(((uint32_t*)array) + j);
						MSB32 = ((MSB32>>31)&1);
						if(MSB32 == 1){
							*(((uint32_t*)array) + j) = 0;
						}
						break;
					case i64: ; 
					case float64: ;
						uint64_t MSB64 = *(((uint64_t*)array) + j);
						MSB64 = ((MSB64>>63)&1);
						if(MSB64 == 1){
							*(((uint64_t*)array) + j) = 0;
						}
						break;
				}
			}
			break;

		case SQUARE : // a = (a)^2
			for(int j=0; j<num_elems; j+=1) {
				switch(a_dt){
					case u8: ; 
						uint8_t val8 = *(((uint8_t*)array) + j);
						val8 *= val8;
						*(((uint8_t*)array) + j) = val8;
						break;

					case u16: ;
						uint16_t val16 = *(((uint16_t*)array) + j);
						val16 *= val16;
						*(((uint16_t*)array) + j) = val16;
						break;

					case u32: ;
						uint32_t val32 = *(((uint32_t*)array) + j);
						val32 *= val32;
						*(((uint32_t*)array) + j) = val32;
						break;

					case u64: ; 
						uint64_t val64 = *(((uint64_t*)array) + j);
						val64 *= val64;
						*(((uint64_t*)array) + j) = val64;
						break;
						
					case i8: ;
						int8_t val8i = *(((int8_t*)array) + j);
						val8i *= val8i;
						*(((int8_t*)array) + j) = val8i;
						break;

					case i16: ;
						int16_t val16i = *(((int16_t*)array) + j);
						val16i *= val16i;
						*(((int16_t*)array) + j) = val16i;
						break;

					case i32: ; 
						int32_t val32i = *(((int32_t*)array) + j);
						val32i *= val32i;
						*(((int32_t*)array) + j) = val32i;
						break;

					case i64: ;
						int64_t val64i = *(((int64_t*)array) + j);
						val64i *= val64i;
						*(((int64_t*)array) + j) = val64i;
						break;

					case float8: ;
						// to be added 
						break;

					case float16: ;
						// to be added 
						break;

					case float32: ;
						float val32f = *(((float*)array) + j);
						val32f *= val32f;
						*(((float*)array) + j) = val32f;
						break;

					case float64: ;
						double val64f = *(((double*)array) + j);
						val64f *= val64f;
						*(((double*)array) + j) = val64f;
						break;
						
				}
			}
			break;

		case EXP : // a = exp(a) // support only for float32 and float64 as of now 
			for(int j=0; j<num_elems; j+=1) {
				switch(a_dt){
					case float8: ;
						// to be added 
						break;

					case float16: ;
						// to be added 
						break;

					case float32: ;
						float x_e32, term_e32;
						x_e32 = *(((float*)array) + j);
						term_e32 = 1;
						float y_e32 = 1;
						for(int iter=1; iter < 1+8; iter+=1) { // Taylor Series approx. upto first 8 terms
							term_e32 *= x_e32/iter;
							y_e32 += term_e32;
						}
						*(((float*)array) + j) = y_e32;
						break;

					case float64: ;
						double x_e64, term_e64;
						x_e64 = *(((double*)array) + j);
						term_e64 = 1;
						double y_e64 = 1;
						for(int iter=1; iter < 1+8; iter+=1) { // Taylor Series approx. upto first 8 terms
							term_e64 *= x_e64/iter;
							y_e64 += term_e64;
						}
						*(((double*)array) + j) = y_e64;
						break;
				}
			}
			break;
		
		case SINE : // a = sin(a) // support only for float32 and float64 as of now 
			for(int j=0; j<num_elems; j+=1) {
				switch(a_dt){
					case float8: ;
						// to be added 
						break;

					case float16: ;
						// to be added 
						break;

					case float32: ;
						float x_s32, term_s32;
						x_s32 = *(((float*)array) + j);
						x_s32 = setrange_float(x_s32); // bring value in (-pi, pi]
						term_s32 = x_s32;
						float y_s32 = x_s32;
						for(int iter=2; iter < 1+8; iter+=1) { // Taylor Series approx. upto first 8 terms
							term_s32 *= -1.0*x_s32*x_s32/(2*iter-1)/(2*iter-2);
							y_s32 += term_s32;
						}
						*(((float*)array) + j) = y_s32;
						break;

					case float64: ;
						double x_s64, term_s64;
						x_s64 = *(((double*)array) + j);
						x_s64 = setrange_double(x_s64); // bring value in (-pi, pi]
						term_s64 = x_s64;
						double y_s64 = x_s64;
						for(int iter=2; iter < 1+8; iter+=1) { // Taylor Series approx. upto first 8 terms
							term_s64 *= -1.0*x_s64*x_s64/(2*iter-1)/(2*iter-2);
							y_s64 += term_s64;
						}
						*(((double*)array) + j) = y_s64;
						break;
				}
			}
			break;

		case ABSOLUTE : // a = |a|
		// for uint types --> i/p = o/p so do nothing 
		// for int types --> check MSB for sign 
		// for float types --> check MSB for sign
			for(int j=0; j<num_elems; j+=1) {
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
	req_a.request_type = WRITE;
	//req_a.write_data = store_here; 
	for(int i=0; i<req_a.arguments[0] ;i=i+1){
		req_a.write_data[i] = store_here[i];
	}
	memPoolAccess(a->mem_pool_identifier,&req_a,&mpr);
	return; 
}
