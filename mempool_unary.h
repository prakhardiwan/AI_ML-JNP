#ifndef _mempool_h____
#define _mempool_h____

//
// mem-pool is organized into 4KB pages.
// Each page consists of 512  64-bit dwords.
//
#define MEMPOOL_WORD_SIZE         8	 // bytes.
#define MEMPOOL_PAGE_SIZE      	  512	 // words
typedef uint64_t MemPoolWord;

typedef struct __MemPoolRequest {
 	// allocate/de-allocate/write/read.
	uint8_t   request_type;
	uint8_t   request_tag;
	uint16_t  argument_0;
	uint32_t  argument_1;
	float*  argument_2;

	// arguments
	//     request_type = allocate
	//          argument_0 = number of pages requested.
	//     request type = deallocate
	//          deallocates page_descriptor in argument 1
	//     read
	//          argument_0 = number of words requested
	//          argument_1 = buffer id
	//     write
	//          argument_0 = number of words requested
	//          argument_1 = buffer id
	//          write_data.
	uint64_t* write_data;
} MemPoolRequest;

typedef struct __MemPoolResponse {
	uint8_t  request_tag;  		// return the request tag.
	uint16_t pool_identifier; 	// which pool?
	uint32_t buffer_id;		// if 0, indicates unsuccessful allocation.
} MemPoolResponse;


/*typedef struct __MemPool {
	uint16_t      mem_pool_index;
	uint32_t      free_list_head;
	MemPoolPage*  page_array;
} MemPool;
*/
// create and initialize mem-pool.
//void createMemPool(uint16_t mem_pool_index, uint32_t mem_pool_size);

// allocate/de-allocate/read/write.
//void memPoolAccess (MemPoolRequest* req, MemPoolResponse* resp);

#endif
