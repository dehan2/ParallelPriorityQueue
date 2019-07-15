#include "ParallelHeapManager.h"
#include "constForParallelHeap.h"
#include <thrust/copy.h>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



////////////////////////////////////////////////////////////////////
// TO DO 190306 - Refer this page, make it possible to use vector in kernel
// https://gist.github.com/docwhite/843f17e33e4c1f2b531a14a4bdfe90ec
////////////////////////////////////////////////////////////////////

__global__ void post_process_for_push(float* heapPtr, InsertItem2* insertUpdateBufPtr, int heapSize, float entity);

__global__ void post_process_for_pop(float* heapPtr, bool* deleteUpdateBufPtr, int heapSize);

__global__ void delete_update2(float* heapPtr, int heapSize, bool* deleteUpdateBufferPtr, bool isOdd);

__device__ void sort_three_entities(float* entities);

__global__ void insert_update2(float* heapPtr, int heapSize, InsertItem2* insertUpdateBufferPtr, bool isOdd);

__device__ int calculate_node_level(int nodeIndex);
__device__ bool is_index_bit_in_number_zero(int number, int index);

__device__ int calculate_assigned_node_index(int num, bool isOdd);


ParallelHeapManager::ParallelHeapManager()
	: m_heapSize(0), m_heapCapacity(INITIAL_HEAP_CAPACITY)
	//m_heap(m_heapCapacity), m_deleteUpdateBuffer(m_heapCapacity), m_insertUpdateBuffer(m_heapCapacity)
{
	m_heap.resize(m_heapCapacity);
	m_deleteUpdateBuffer.resize(m_heapCapacity);
	m_insertUpdateBuffer.resize(m_heapCapacity);

	initialize_heap_and_buffer();
	
#ifdef USE_PINNED_BUFFER_CHECK
	cudaHostAlloc((void**)&m_bInsertUpdateBufferEmpty, sizeof(bool), cudaHostAllocMapped);
	cudaHostAlloc((void**)&m_bDeleteUpdateBufferEmpty, sizeof(bool), cudaHostAllocMapped);
#endif
}



void ParallelHeapManager::push(const float& entity)
{
	//It is guaranteed that there is no pop operation until every item is inserted
	/*InsertItem item;
	item.currNode = 0;
	item.destination = m_heap.size();
	item.entity = entity;
	m_insertUpdateBuffer_evenLevel.push_back(item);
	m_heap.push_back(INT_MAX);*/

	float* heapPtr = raw_pointer_cast(m_heap.data());
	InsertItem2* insertUpdateBufPtr = raw_pointer_cast(m_insertUpdateBuffer.data());
	increase_heap_size();
	post_process_for_push<<<1, 1 >>>(heapPtr, insertUpdateBufPtr, m_heapSize, entity);
	//std::cout << "Push: " << entity << " - destination: " << m_heapSize - 1 << std::endl;
}



float ParallelHeapManager::pop()
{
	float topEntity = m_heap.front();
	//std::cout << "Pop: " << topEntity << std::endl;

	/*m_deleteUpdateBuffer_evenLevel.push_back(0);
	m_heap.front() = m_heap.back();
	m_heap.pop_back();*/
	float* heapPtr = raw_pointer_cast(m_heap.data());
	bool* deleteUpdateBufPtr = raw_pointer_cast(m_deleteUpdateBuffer.data());

	post_process_for_pop<<<1,1>>>(heapPtr, deleteUpdateBufPtr, m_heapSize);
	m_heapSize--;

	return topEntity;
}



void ParallelHeapManager::increase_heap_size()
{
	if (m_heapSize == m_heapCapacity)
	{
		//std::cout << "Heap capacity increase" << std::endl;
		m_heapCapacity *= 2;
		m_heap.resize(m_heapCapacity);
		m_deleteUpdateBuffer.resize(m_heapCapacity);
		m_insertUpdateBuffer.resize(m_heapCapacity);

		initialize_heap_and_buffer();
	}
	m_heapSize++;
	//std::cout << "Heap size increased: " << m_heapSize << ", heap capacity: " << m_heapCapacity << std::endl;
}



void ParallelHeapManager::initialize_heap_and_buffer()
{
	thrust::fill(m_heap.begin() + m_heapSize, m_heap.end(), -1.0f);
	thrust::fill(m_deleteUpdateBuffer.begin() + m_heapSize, m_deleteUpdateBuffer.end(), false);
	thrust::fill(m_insertUpdateBuffer.begin() + m_heapSize, m_insertUpdateBuffer.end(), InsertItem2{ -1, -1.0f });
}



void ParallelHeapManager::initiate_delete_update(const bool& isOdd)
{
	float* heapPtr = raw_pointer_cast(m_heap.data());
	bool* deleteUpdateBufPtr = raw_pointer_cast(m_deleteUpdateBuffer.data());
	delete_update2<<<1, NT >>>(heapPtr, m_heapSize, deleteUpdateBufPtr, isOdd);
}



__global__ void post_process_for_pop(float* heapPtr, bool* deleteUpdateBufPtr, int heapSize)
{
	heapPtr[0] = heapPtr[heapSize-1];
	deleteUpdateBufPtr[0] = true;

	heapPtr[heapSize - 1] = -1.0f;
	deleteUpdateBufPtr[heapSize - 1] = false;
}



__global__ void post_process_for_push(float* heapPtr, InsertItem2* insertUpdateBufPtr, int heapSize, float entity)
{
	heapPtr[heapSize - 1] = FLT_MAX;
	insertUpdateBufPtr[0].destination = heapSize-1;
	insertUpdateBufPtr[0].entity = entity;
}



__global__ void delete_update2(float* heapPtr, int heapSize, bool* deleteUpdateBufferPtr, bool isOdd)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	//clock_t start_time = clock();
	int currIndex = calculate_assigned_node_index(tid, isOdd);
	//clock_t stop_time = clock();
	//int cycle1 = (int)(stop_time - start_time);

	//int cycle2 = 0;
	//int cycle3 = 0;

	while (currIndex < heapSize)
	{
		//start_time = clock();
		bool bNeedUpdate = deleteUpdateBufferPtr[currIndex];
		//stop_time = clock();
		//cycle2 += (int)(stop_time - start_time);

		if (bNeedUpdate == true)
		{
			//printf("Delete update: [%d, %f]\n", currIndex, heapPtr[currIndex]);
			int leftIndex = 2 * currIndex + 1;
			int rightIndex = 2 * currIndex + 2;
			if (leftIndex < heapSize)
			{
				if (rightIndex < heapSize)
				{
					//All three nodes exist.

					float sortingBuffer[3];
					sortingBuffer[0] = heapPtr[currIndex];
					sortingBuffer[1] = heapPtr[leftIndex];
					sortingBuffer[2] = heapPtr[rightIndex];
					//printf("Sorting - [%d, %f], [%d, %f], [%d, %f]\n", currIndex, heapPtr[currIndex], leftIndex, heapPtr[leftIndex], rightIndex, heapPtr[rightIndex]);

					sort_three_entities(sortingBuffer);

					heapPtr[currIndex] = sortingBuffer[0];
					if (heapPtr[leftIndex] < heapPtr[rightIndex])
					{
						heapPtr[leftIndex] = sortingBuffer[2];
						heapPtr[rightIndex] = sortingBuffer[1];
						deleteUpdateBufferPtr[leftIndex] = true;
					}
					else
					{
						heapPtr[leftIndex] = sortingBuffer[1];
						heapPtr[rightIndex] = sortingBuffer[2];
						deleteUpdateBufferPtr[rightIndex] = true;
					}
				}
				else
				{
					//Only left node exists. - It has no child
					if (heapPtr[currIndex] > heapPtr[leftIndex])
					{
						float temp = heapPtr[currIndex];
						heapPtr[currIndex] = heapPtr[leftIndex];
						heapPtr[leftIndex] = temp;
					}
				}
			}
			deleteUpdateBufferPtr[currIndex] = false;
		}
		tid += NT;

		//start_time = clock();
		currIndex = calculate_assigned_node_index(tid, isOdd);
		//stop_time = clock();
		//cycle3 += (int)(stop_time - start_time);
	}

	//printf("Delete update time: %d, %d, %d\n", cycle1, cycle2, cycle3);
}




__device__ void sort_three_entities(float* entities)
{
	if (entities[0] > entities[2])
	{
		float temp = entities[0];
		entities[0] = entities[2];
		entities[2] = temp;
	}

	if (entities[0] > entities[1])
	{
		float temp = entities[0];
		entities[0] = entities[1];
		entities[1] = temp;
	}
	
	if (entities[1] > entities[2])
	{
		float temp = entities[1];
		entities[1] = entities[2];
		entities[2] = temp;
	}
}



void ParallelHeapManager::initiate_insert_update(const bool& isOdd)
{
	//std::cout << "Start insert update: " << isOdd << std::endl;
	float* heapPtr = raw_pointer_cast(m_heap.data());
	InsertItem2* insertUpdateBufPtr = raw_pointer_cast(m_insertUpdateBuffer.data());
	insert_update2<<<1, NT >>>(heapPtr, m_heapSize, insertUpdateBufPtr, isOdd);
	//cudaDeviceSynchronize();
	//std::cout << "Finish insert update: " << std::endl;
}



__global__ void insert_update2(float* heapPtr, int heapSize, InsertItem2* insertUpdateBufferPtr, bool isOdd)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int currIndex = calculate_assigned_node_index(tid, isOdd);

	while (currIndex < heapSize)
	{
		InsertItem2* currItem = &insertUpdateBufferPtr[currIndex];
		if (currItem->destination >= 0)
		{
			//printf("insert update: %d - [%d, %f] \n", currIndex, currItem->destination, currItem->entity);
			if (currIndex == currItem->destination)
			{
				//printf("Arrive\n");

				heapPtr[currIndex] = currItem->entity;
			}
			else
			{
				//printf("Not Arrived\n");

				float entityForNextLevel = currItem->entity;
				if (heapPtr[currIndex] > entityForNextLevel)
				{
					entityForNextLevel = heapPtr[currIndex];
					heapPtr[currIndex] = currItem->entity;
				}

				int levelOfTargetNode = calculate_node_level(currItem->destination);
				int levelOfCurrNode = calculate_node_level(currIndex);
				bool goLeft = is_index_bit_in_number_zero(currItem->destination + 1, (levelOfTargetNode - levelOfCurrNode));

				int nextIndex = 0;
				if (goLeft)
					nextIndex = 2 * currIndex + 1;
				else
					nextIndex = 2 * currIndex + 2;

				InsertItem2* nextItem = &insertUpdateBufferPtr[nextIndex];
				nextItem->destination = currItem->destination;
				nextItem->entity = entityForNextLevel;

				//printf("Next item: [%d, %d, %f]\n", nextIndex, nextItem->destination, nextItem->entity);
			}

			currItem->destination = -1;
			currItem->entity = -1.0f;
		}
		tid += NT;
		currIndex = calculate_assigned_node_index(tid, isOdd);
	}
}



__device__ int calculate_node_level(int nodeIndex)
{
	int level = 1;
	int maxLevelIndex = 0;
	int powerOfTwo = 1;

	while (nodeIndex > maxLevelIndex)
	{
		powerOfTwo = (powerOfTwo<<1); //powerOfTwo *= 2;
		maxLevelIndex += powerOfTwo;
		level++;
	}

	return level;
}



__device__ bool is_index_bit_in_number_zero(int number, int index)
{
	int translatedNum = number >> (index - 1);
	int result = translatedNum % 2;
	return result == 0;
}



//Frontline 190318 - ¿Ï¼ºÇÏÀð!
__device__ int calculate_assigned_node_index(int num, bool isOdd)
{
	num += 1;
	int level = 0;
	if (isOdd)
		level = 1;

	int numEntity = 1<<level; //pow(2, level)

	while (num > numEntity)
	{
		level += 2;
		numEntity += 1 << level; //pow(2, level)
	}
	
	int lastIndex = 0;
	for (int i = 1; i <= level; i++)
	{
		lastIndex += 1 << i; //pow(2, i)
	}
	int targetIndex = lastIndex - numEntity + num;
	
	//printf("targetIndexCalculation: [%d, %d, %d, %d, %d, %d]\n", isOdd, num, level, numEntity, lastIndex, targetIndex);

	return targetIndex;
}



//__device__ bool d_isDeleteUpdateBufferEmpty;

//__device__ bool d_isInsertUpdateBufferEmpty;


__global__ void check_delete_update_buffer_empty(bool* deleteUpdateBufferPtr, int heapSize, bool* result)
{
	*result = true;
	for (int i = 0; i < heapSize; i++)
	{
		if (deleteUpdateBufferPtr[i] == true)
		{
			*result = false;
			break;
		}
	}
}



__global__ void check_insert_update_buffer_empty(InsertItem2* insertUpdateBufferPtr, int heapSize, bool* result)
{
	*result = true;
	for (int i = 0; i < heapSize; i++)
	{
		if (insertUpdateBufferPtr[i].destination >= 0)
		{
			//printf("Insert buffer is not empty:[%d, %d, %f]\n", i, insertUpdateBufferPtr[i].destination, insertUpdateBufferPtr[i].entity);
			*result = false;
			break;
		}
	}
}



bool ParallelHeapManager::is_delete_update_buffer_empty()
{
	bool*  deleteUpdateBufPtr = raw_pointer_cast(m_deleteUpdateBuffer.data());
	bool* d_result;

#ifndef USE_PINNED_BUFFER_CHECK
	bool h_result;
	cudaMalloc(&d_result, sizeof(bool));
	check_delete_update_buffer_empty<<<1, 1>>>(deleteUpdateBufPtr, m_heapSize, d_result);
	//cudaDeviceSynchronize();
	
	cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(d_result);
	return h_result;
#else
	cudaHostGetDevicePointer((void**)&d_result, (void*)m_bDeleteUpdateBufferEmpty, 0);
	check_delete_update_buffer_empty << <1, 1 >> > (deleteUpdateBufPtr, m_heapSize, d_result);

	return *m_bDeleteUpdateBufferEmpty;
#endif

	//bool isDeleteUpdateBufferEmpty = false;
	//cudaMemcpyFromSymbol(&isDeleteUpdateBufferEmpty, d_isDeleteUpdateBufferEmpty, sizeof(d_isDeleteUpdateBufferEmpty), 0, cudaMemcpyDeviceToHost);
	//return isDeleteUpdateBufferEmpty;

	
}




bool ParallelHeapManager::is_insert_update_buffer_empty()
{
	InsertItem2* insertUpdateBufPtr = raw_pointer_cast(m_insertUpdateBuffer.data());
	bool* d_result;

#ifndef USE_PINNED_BUFFER_CHECK
	bool h_result;
	cudaMalloc(&d_result, sizeof(bool));
	check_insert_update_buffer_empty<<<1,1>>>(insertUpdateBufPtr, m_heapSize, d_result);
	//cudaDeviceSynchronize();

	cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(d_result);
	return h_result;
#else
	cudaHostGetDevicePointer((void**)&d_result, (void*)m_bInsertUpdateBufferEmpty, 0);
	check_insert_update_buffer_empty << <1, 1 >> > (insertUpdateBufPtr, m_heapSize, d_result);

	return *m_bInsertUpdateBufferEmpty;
#endif

	//bool isInsertUpdateBufferEmpty = false;
	//cudaMemcpyFromSymbol(&isInsertUpdateBufferEmpty, d_isInsertUpdateBufferEmpty, sizeof(d_isInsertUpdateBufferEmpty), 0, cudaMemcpyDeviceToHost);
	//return isInsertUpdateBufferEmpty;
}


__global__ void print_heap_kernel(float* heapPtr, int heapSize)
{
	for (int i = 0; i<heapSize; i++)
	{
		printf("Heap[%d]: %f\n", i, heapPtr[i]);
	}
}



void ParallelHeapManager::print_heap()
{
	std::cout << "Print heap: " <<m_heapSize<<std::endl;
	float* heapPtr = raw_pointer_cast(m_heap.data());
	print_heap_kernel<<<1,1>>>(heapPtr, m_heapSize);
	cudaDeviceSynchronize();
	std::cout << std::endl << std::endl;
}


