#include<stdio.h>



void binaryOperatorOnTensor (Tensor* a, Tensor* b, Operation op){
	// In-place binary operator: performs a = a op b 
	// supported op --> +,-,*,/ 
	

	// Possible issue: What if memory regions of the 2 tensors overlap : check condition required.. 

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

	// Memory acccess; 


	}