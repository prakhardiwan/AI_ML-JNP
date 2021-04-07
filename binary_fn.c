#include<stdio.h>

#define BLOCK_SIZE 8;


// Cache
uint32_t cache1[BLOCK_SIZE];
uint32_t cache2[BLOCK_SIZE];

void readMemPool(uint32_t cache[], MemPoolRequest* req){
	uint32_t ptr = req.argument_2;
	for(int i=0;i<BLOCK_SIZE;i=i+1){
		cache[i] = *(ptr+i);		
	}
}

void writeMemPool(uint32_t cache[], MemPoolRequest* req){
	uint32_t ptr = req.argument_2;
	for(int i=0;i<BLOCK_SIZE;i=i+1){
		*(ptr+i) = cache[i] ;		
	}
}

void binaryOperatorOnTensor (Tensor* a, Tensor* b, char op){
	// In-place binary operator: performs a = a op b : element-wise
	// supported op --> +,-,*,/ 

	// $ Possible issue: What if memory regions of the 2 tensors overlap : check condition required.. 
	// --> Assumed that non-overlapping memory regions 

	// $ Possible issue: Typecasting required for hw model || Universal Cache for all data types
	// --> Assumed datatype 

	// $ Possible issue: cache structure defn
	// --> we have assumed 2D for operation; and one dimension = BLOCK_SIZE

	// $ Possible issue: 1 vector in row-major format and another vector col major format 
	// [a a]
	// [b b]
	// [c c] ---> in row-major rep
	// [a]
	// [a]
	// [b]
	// [b]
	// [c]
	// [c]

	// store tensor descriptors in local memory declared descriptors  
	TensorDescriptor td_a = a.descriptor;
	TensorDescriptor td_b = b.descriptor;
	// Number of dimensions
	uint32_t a_num_dim = td_a.number_of_dimensions;
	uint32_t b_num_dim = td_b.number_of_dimensions;

	// Total iterations
	uint32_t max_iter = 1;
	// Tensor dimensions match check 
	if(a_num_dim!= b_num_dim){
		printf("Tensors incompatible: Number of dimensions don't match");
		return;
	}
	else{
		int n_dim = td_a.number_of_dimensions();
		for(int i=0; i<n_dim; i=i+1){
			if(td_a.dimensions[i]!=td_b.dimensions[i]){
				printf("Tensors incompatible: Atleast one of the dimensions %d don't match", i);
				return;
			}
			max_iter *= td_a.dimensions[i]; 
		}
		max_iter /= BLOCK_SIZE; 
		if(max_iter==0){
			max_iter = 1;
		}
	}

	// Memory acccess: can be block wise OR element by element 
	// $ here we do block-wise 

	// $ Make sure no memory overlap is present amongst the tensor memories

	// performing operations: 
	// $ we can check if op is supported or not before memory access for saving time 
	

	switch(op){ // for request_type: read --> 2; write -->3
		case '+' : // a = a + b
			for(int i=0; i<max_iter; i=i+1){
				MemPoolRequest* req_a;
				req_a.request_type = 2;
				req_a.request_tag = i; //
				req_a.argument_0 = BLOCK_SIZE; //assumed datatype/size
				req_a.argument_1 = a.mem_pool_identifier;
				req_a.argument_2 = a.mem_pool_buffer_pointer + BLOCK_SIZE*i; // data size to be incorporated

				MemPoolRequest* req_b;
				req_b.request_type = 2;
				req_b.request_tag = i; //
				req_b.argument_0 = BLOCK_SIZE; //assumed datatype/size
				req_b.argument_1 = b.mem_pool_identifier;
				req_b.argument_2 = b.mem_pool_buffer_pointer + BLOCK_SIZE*i;

				readMemPool(cache1,req_a);	
				readMemPool(cache2,req_b);
				
				for(int j=0; j<BLOCK_SIZE; j=j+1){ //typecasting problem
					cache1[j] = cache1[j] + cache2[j];
				}
				req_a.request_type = 3;
				writeMemPool(cache1,req_a);
			}
			break;
		case '-' : 
			break;

		case '*' :
			break;

		case '/' : 
			break;

		default  : 
			printf("%c operation not supported",op);
			return;
	}

}

