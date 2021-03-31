#include<stdio.h>

#define BLOCK_SIZE 8;

void binaryOperatorOnTensor (Tensor* a, Tensor* b, char op){
	// In-place binary operator: performs a = a op b : element-wise
	// supported op --> +,-,*,/ 

	// $ Possible issue: What if memory regions of the 2 tensors overlap : check condition required.. 

	// store tensor descriptors in local memory declared descriptors  
	TensorDescriptor td_a = a.descriptor;
	TensorDescriptor td_b = b.descriptor;
	
	// Tensor dimensions match check 
	if(td_a.number_of_dimensions != td_b.number_of_dimensions){
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
		}
	}

	// Memory acccess: can be block wise OR element by element 
	// $ here we do block-wise 

	// $ Make sure no memory overlap is present amongst the tensor memories
	


	// performing operations: 
	// $ we can check if op is supported or not before memory access for saving time 
	switch(op){
		case '+' : // a = a + b

		case '-' : 

		case '*' :

		case '/' : 

		default  : 
			printf("%c operation not supported",op);
			return;
	}

}
