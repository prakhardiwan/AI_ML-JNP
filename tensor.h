#ifndef __tensor_h___
#define __tensor_h___

#define MAX_DIMENSIONS 64

typedef enum __TensorDataType {
	u8, u16, u32, u64,
	i8, i16, i32, i64,
	float16, float32, float64
} TensorDataType;

typedef struct __TensorDescriptor {

	TensorDataType data_type;


	// data can be in row-major form
	// or column major form.
	//
	// row-major [0][0], [0][1] etc...
	// column-major [0][0], [1][0] etc...
	uint8_t row_major_form;

	uint32_t number_of_dimensions;
	uint32_t dimensions[MAX_DIMENSIONS];

} TensorDescriptor;


typedef struct __Tensor{
	TensorDescriptor descriptor;

	// chain of buffers will hold data in either
	// row major or column major form as indicated
	// in the descriptor.
	uint16_t mem_pool_identifier;
	uint32_t* mem_pool_buffer_pointer;
} Tensor;


// start
//void createTensor (uint32_t ndim, uint32_t* dims, TensorDataType dt, uint16_t mempool);


#endif
