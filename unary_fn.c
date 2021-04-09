#define BLOCK_SIZE 8 // bring in these many tensor elements at a time in local cache

// cache
float cache[BLOCK_SIZE];

void readMemPool(float cache[], MemPoolRequest* req) {
	uint32_t* ptr = req->argument_2;
	for(int i=0; i<BLOCK_SIZE; i+=1) {
		cache[i] = *(ptr + 4*i);
	}
}

void writeMemPool(float cache[], MemPoolRequest* req) {
	uint32_t* ptr = req->argument_2;
	for(int i=0; i<BLOCK_SIZE; i+=1) {
		*(ptr + 4*i) = cache[i];
	}
}

float absolute(float x) {

	if(x >= 0) { return x;}
	
	return (-1.0) * x;
}

void unaryOperatorOnTensor(Tensor* a, char op) {
	// in-place unary operator: performs a = f(a) where f is specified by op
	// supported op --> s(for sine), e(for exp), r(for RELU), 2(for square), a(for absolute)

	TensorDescriptor td_a = a->descriptor;

	uint32_t num_dim = td_a.number_of_dimensions; // number of dimensions

	uint32_t flat_dims = 1; // product of dims (# of elements in tensor)
	uint32_t max_iter = 1;  // # of iterations over which cache loaded, op performed, data sent back

	for(uint32_t i=0; i<n_dim; i+=1) {

		flat_dims *= td_a.dimensions[i];
	}

	max_iter = ceil( ((float)flat_dims) / BLOCK_SIZE);
    uint32_t dims = 0;
    uint32_t k;

	switch(op){ // for request_type: read --> 2; write -->3
		case 'r' : // a = RELU(a)

			for(int i=0; i<max_iter; i+=1) {

				MemPoolRequest req_a;
				req_a.request_type = 2;
				req_a.request_tag = i; //
				req_a.argument_0 = BLOCK_SIZE; //assumed datatype/size
				req_a.argument_1 = a->mem_pool_identifier;
				req_a.argument_2 = (a->mem_pool_buffer_pointer) + BLOCK_SIZE*i*4; // assumption: byte-adr mem

				readMemPool(cache, &req_a);

                if(dims + BLOCK_SIZE > flat_dims) { k = flat_dims - dims;}
                else { k = BLOCK_SIZE;}

				for(int j=0; j<k; j+=1) {
					if(cache[j]<0) { cache[j]=0;}
					else { continue;}
				}

				req_a.request_type = 3;
				writeMemPool(cache, &req_a);

				dims += k;
			}

			break;
		
		case '2' : // a = (a)^2

			for(int i=0; i<max_iter; i+=1) {

				MemPoolRequest req_a;
				req_a.request_type = 2;
				req_a.request_tag = i; //
				req_a.argument_0 = BLOCK_SIZE; //assumed datatype/size
				req_a.argument_1 = a->mem_pool_identifier;
				req_a.argument_2 = (a->mem_pool_buffer_pointer) + BLOCK_SIZE*i*4; // assumption: byte-adr mem

				readMemPool(cache, &req_a);

                if(dims + BLOCK_SIZE > flat_dims) { k = flat_dims - dims;}
                else { k = BLOCK_SIZE;}

				for(int j=0; j<k; j+=1) {
					cache[j] *= cache[j];
				}

				req_a.request_type = 3;
				writeMemPool(cache, &req_a);

				dims += k;
			}

			break;

		case 's' : // a = sin(a)
			float eps = 1.0e-6;
            float x, term;
            uint32_t MAX = 2 + (1e5); // max iterations 
			
			for(int i=0; i<max_iter; i+=1) {

				MemPoolRequest req_a;
				req_a.request_type = 2;
				req_a.request_tag = i; //
				req_a.argument_0 = BLOCK_SIZE; //assumed datatype/size
				req_a.argument_1 = a->mem_pool_identifier;
				req_a.argument_2 = (a->mem_pool_buffer_pointer) + BLOCK_SIZE*i*4; // assumption: byte-adr mem

				readMemPool(cache, &req_a);

                if(dims + BLOCK_SIZE > flat_dims) { k = flat_dims - dims;}
                else { k = BLOCK_SIZE;}
				
				for(int j=0; j<k; j+=1) {

	                x = cache[j];
					term = cache[j];

					for(int iter=2; absolute(term) > eps; iter+=1) {
						if(iter == MAX) {
							cout << "MAX_ITERS exceeded for sine subroutine. Tensor element(s) too large.";
							return;
						}

						term *= -1.0*x*x/(2*k-1)/(2*k-2);
						cache[j] += term;
					}
				}

				req_a.request_type = 3;
				writeMemPool(cache, &req_a);

				dims += k;
			}

			break;

		case '/' :
			break;

		default  :
			printf("%c operation not supported",op);
			
			return;
	}

}
