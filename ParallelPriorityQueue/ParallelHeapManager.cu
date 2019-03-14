#include "ParallelHeapManager.h"
#include "constForParallelHeap.h"
#include <thrust/copy.h>
#include <iostream>
#include "cuda.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"



////////////////////////////////////////////////////////////////////
// TO DO 190306 - Refer this page, make it possible to use vector in kernel
// https://gist.github.com/docwhite/843f17e33e4c1f2b531a14a4bdfe90ec
////////////////////////////////////////////////////////////////////

__global__ void post_process_for_push(InsertItem2* insertUpdateBufPtr, int heapSize, float entity);

__global__ void post_process_for_pop(float* heapPtr, bool* deleteUpdateBufPtr, int heapSize);

/*__global__ void delete_update(float* heapPtr, int heapSize,
	int* deleteUpdateBufferPtr_current, int bufferSize
	, int* deleteUpdateBufferPtr_next);*/

__global__ void delete_update2(float* heapPtr, int heapSize, bool* deleteUpdateBufferPtr, bool isOdd);

__device__ void sort_three_entities(float* entities);


/*__global__ void insert_update(float* heapPtr, int heapSize,
	InsertItem* insertUpdateBufferPtr_current, int bufferSize
	, InsertItem* insertUpdateBufferPtr_next);*/

__global__ void insert_update2(float* heapPtr, int heapSize, InsertItem2* insertUpdateBufferPtr, bool isOdd);

__device__ int calculate_node_level(int nodeIndex);
__device__ bool is_index_bit_in_number_zero(int number, int index);



ParallelHeapManager::ParallelHeapManager()
	: m_heapSize(0), m_heapCapacity(INITIAL_HEAP_CAPACITY)
	//m_heap(m_heapCapacity), m_deleteUpdateBuffer(m_heapCapacity), m_insertUpdateBuffer(m_heapCapacity)
{
	m_heap.resize(m_heapCapacity);
	m_deleteUpdateBuffer.resize(m_heapCapacity);
	m_insertUpdateBuffer.resize(m_heapCapacity);

	initialize_heap_and_buffer();
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

	InsertItem2* insertUpdateBufPtr = raw_pointer_cast(m_insertUpdateBuffer.data());
	post_process_for_push<<<1, 1 >>>(insertUpdateBufPtr, m_heapSize, entity);
	increase_heap_size();
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
		std::cout << "Heap capacity increase" << std::endl;
		m_heapCapacity *= 2;
		m_heap.resize(m_heapCapacity);
		m_deleteUpdateBuffer.resize(m_heapCapacity);
		m_insertUpdateBuffer.resize(m_heapCapacity);

		initialize_heap_and_buffer();
	}
	m_heapSize++;
	std::cout << "Heap size increased: " << m_heapSize << ", heap capacity: " << m_heapCapacity << std::endl;
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
	delete_update2 <<<1, NT >>> (heapPtr, m_heap.size(), deleteUpdateBufPtr, isOdd);
	
	/*device_vector<int> tempBuffer_device;
	int bufferSize = 0;

	if (isOdd)
	{
		//printf("Insert update - Odd: %d \n", m_insertUpdateBuffer_oddLevel.size());
		if (!m_deleteUpdateBuffer_oddLevel.empty())
		{
			bufferSize = m_deleteUpdateBuffer_oddLevel.size();
			int* deleteUpdateBufferPtr = raw_pointer_cast(m_deleteUpdateBuffer_oddLevel.data());

			tempBuffer_device.resize(bufferSize);
			int* tempBufferPtr = raw_pointer_cast(tempBuffer_device.data());

			delete_update <<<1, NT >> > (heapPtr, m_heap.size(),
				deleteUpdateBufferPtr, bufferSize, tempBufferPtr);

			m_deleteUpdateBuffer_oddLevel.clear();
		}
	}
	else
	{
		//printf("Insert update - Even: %d \n", m_insertUpdateBuffer_evenLevel.size());
		if (!m_deleteUpdateBuffer_evenLevel.empty())
		{
			bufferSize = m_deleteUpdateBuffer_evenLevel.size();
			int* deleteUpdateBufferPtr = raw_pointer_cast(m_deleteUpdateBuffer_evenLevel.data());

			tempBuffer_device.resize(bufferSize);
			int* tempBufferPtr = raw_pointer_cast(tempBuffer_device.data());

			delete_update << <1, NT >> > (heapPtr, m_heap.size(),
				deleteUpdateBufferPtr, bufferSize, tempBufferPtr);

			m_deleteUpdateBuffer_evenLevel.clear();
		}
	}


	if (bufferSize > 0)
	{
		host_vector<int> tempBuffer_host(tempBuffer_device.begin(), tempBuffer_device.end());
		auto it = tempBuffer_host.begin();
		while (it != tempBuffer_host.end())
		{
			if (*it == -1)
				it = tempBuffer_host.erase(it);
			else
				it++;
		}

		if (isOdd)
		{
			for (auto& item : tempBuffer_host)
				m_deleteUpdateBuffer_evenLevel.push_back(item);
		}
		else
		{
			for (auto& item : tempBuffer_host)
				m_deleteUpdateBuffer_oddLevel.push_back(item);
		}
	}*/
}



__global__ void post_process_for_pop(float* heapPtr, bool* deleteUpdateBufPtr, int heapSize)
{
	heapPtr[0] = heapPtr[heapSize-1];
	deleteUpdateBufPtr[0] = true;

	heapPtr[heapSize - 1] = -1.0f;
	deleteUpdateBufPtr[heapSize - 1] = false;
}



__global__ void post_process_for_push(InsertItem2* insertUpdateBufPtr, int heapSize, float entity)
{
	InsertItem2& bufFront = insertUpdateBufPtr[0];
	bufFront.destination = heapSize;
	bufFront.entity = entity;
}



/*__global__ void delete_update(float* heapPtr, int heapSize,
	int* deleteUpdateBufferPtr_current, int deleteUpdateBufferSize
	, int* deleteUpdateBufferPtr_next)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < deleteUpdateBufferSize)
	{
		int currIndex = deleteUpdateBufferPtr_current[tid];
		deleteUpdateBufferPtr_next[tid] = -1;
		if (currIndex < heapSize)
		{
			int leftIndex = 2 * currIndex + 1;
			int rightIndex = 2 * currIndex + 2;
			if (leftIndex < heapSize)
			{
				if (rightIndex < heapSize)
				{
					//All three nodes exist.
					
					float sortingBuffer[3];
					//cudaMalloc((void**)&sortingBuffer, 3 * sizeof(float));
					sortingBuffer[0] = heapPtr[currIndex];
					sortingBuffer[1] = heapPtr[leftIndex];
					sortingBuffer[2] = heapPtr[rightIndex];
					

					sort_three_entities(sortingBuffer);
					
					heapPtr[currIndex] = sortingBuffer[0];
					if (heapPtr[leftIndex] < heapPtr[rightIndex])
					{
						heapPtr[leftIndex] = sortingBuffer[2];
						heapPtr[rightIndex] = sortingBuffer[1];
						deleteUpdateBufferPtr_next[tid] = leftIndex;
					}
					else
					{
						heapPtr[leftIndex] = sortingBuffer[1];
						heapPtr[rightIndex] = sortingBuffer[2];
						deleteUpdateBufferPtr_next[tid] = rightIndex;
					}

					//cudaFree(sortingBuffer);
				}
				else
				{
					//Only left node exists.
					if (heapPtr[currIndex] > heapPtr[leftIndex])
					{
						float temp = heapPtr[currIndex];
						heapPtr[currIndex] = heapPtr[rightIndex];
						heapPtr[rightIndex] = temp;
					}
				}
			}
			else
			{
				//Only current node exists.british highlander
				//Do nothing
			}
		}
		tid += NT;
	}
}
*/


__global__ void delete_update2(float* heapPtr, int heapSize, bool* deleteUpdateBufferPtr, bool isOdd)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int currIndex = 0;
	if (isOdd)
		currIndex = 2 * tid + 1;
	else
		currIndex = 2 * tid;

	while (currIndex < heapSize)
	{
		if (deleteUpdateBufferPtr[currIndex] == true)
		{
			int leftIndex = 2 * currIndex + 1;
			int rightIndex = 2 * currIndex + 2;
			if (leftIndex < heapSize)
			{
				if (rightIndex < heapSize)
				{
					//All three nodes exist.

					float sortingBuffer[3];
					//cudaMalloc((void**)&sortingBuffer, 3 * sizeof(float));
					sortingBuffer[0] = heapPtr[currIndex];
					sortingBuffer[1] = heapPtr[leftIndex];
					sortingBuffer[2] = heapPtr[rightIndex];

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
						heapPtr[currIndex] = heapPtr[rightIndex];
						heapPtr[rightIndex] = temp;
					}
				}
			}
			deleteUpdateBufferPtr[currIndex] = false;
		}
		tid += NT;
	}
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
	float* heapPtr = raw_pointer_cast(m_heap.data());
	InsertItem2* insertUpdateBufPtr = raw_pointer_cast(m_insertUpdateBuffer.data());
	insert_update2 <<<1, NT >>> (heapPtr, m_heap.size(), insertUpdateBufPtr, isOdd);
}


/*
__global__ void insert_update(float* heapPtr, int heapSize,
	InsertItem* insertUpdateBufferPtr_current, int insertUpdateBufferSize
	, InsertItem* insertUpdateBufferPtr_next)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	while (tid < insertUpdateBufferSize)
	{
		InsertItem currItem = insertUpdateBufferPtr_current[tid];
		//printf("insert update: %d - [%d, %d, %f] \n", tid, currItem.currNode, currItem.destination, currItem.entity);

		if (currItem.currNode == currItem.destination)
		{
			//printf("Arrive\n");

			heapPtr[currItem.destination] = currItem.entity;

			InsertItem dummyItem;
			dummyItem.currNode = -1;
			dummyItem.destination = -1;
			dummyItem.entity = -1;
			insertUpdateBufferPtr_next[tid] = dummyItem;
		}
		else
		{
			//printf("Not Arrived\n");

			float entityForNextLevel = currItem.entity;
			if (heapPtr[currItem.currNode] > entityForNextLevel)
			{
				entityForNextLevel = heapPtr[currItem.currNode];
				heapPtr[currItem.currNode] = currItem.entity;
			}

			int levelOfTargetNode = calculate_node_level(currItem.destination);
			int levelOfCurrNode = calculate_node_level(currItem.currNode);
			bool goLeft = is_index_bit_in_number_zero(currItem.destination + 1, (levelOfTargetNode - levelOfCurrNode));

			InsertItem nextItem;
			nextItem.destination = currItem.destination;
			nextItem.entity = entityForNextLevel;
			if (goLeft)
				nextItem.currNode = 2 * currItem.currNode + 1;
			else
				nextItem.currNode = 2 * currItem.currNode + 2;

			//printf("Next item: [%d, %d, %f]\n", nextItem.currNode, nextItem.destination, nextItem.entity);

			insertUpdateBufferPtr_next[tid] = nextItem;
		}
		tid += NT;
	}
}
*/

__global__ void insert_update2(float* heapPtr, int heapSize, InsertItem2* insertUpdateBufferPtr, bool isOdd)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int currIndex = 0;
	if (isOdd)
		currIndex = 2 * tid + 1;
	else
		currIndex = 2 * tid;

	while (currIndex < heapSize)
	{
		InsertItem2& currItem = insertUpdateBufferPtr[currIndex];
		printf("insert update: %d - [%d, %f] \n", tid, currItem.destination, currItem.entity);
		if (currItem.destination >= 0)
		{
			if (currIndex == currItem.destination)
			{
				printf("Arrive\n");

				heapPtr[currIndex] = currItem.entity;
			}
			else
			{
				printf("Not Arrived\n");

				float entityForNextLevel = currItem.entity;
				if (heapPtr[currIndex] > entityForNextLevel)
				{
					entityForNextLevel = heapPtr[currIndex];
					heapPtr[currIndex] = currItem.entity;
				}

				int levelOfTargetNode = calculate_node_level(currItem.destination);
				int levelOfCurrNode = calculate_node_level(currIndex);
				bool goLeft = is_index_bit_in_number_zero(currItem.destination + 1, (levelOfTargetNode - levelOfCurrNode));

				int nextIndex = 0;
				if (goLeft)
					nextIndex = 2 * currIndex + 1;
				else
					nextIndex = 2 * currIndex + 2;

				InsertItem2& nextItem = insertUpdateBufferPtr[nextIndex];
				nextItem.destination = currItem.destination;
				nextItem.entity = entityForNextLevel;

				printf("Next item: [%d, %f]\n", nextItem.destination, nextItem.entity);
			}

			currItem.destination = -1;
			currItem.entity = -1.0f;
		}
		tid += NT;
	}
}



__device__ int calculate_node_level(int nodeIndex)
{
	int level = 1;
	int maxLevelIndex = 0;
	int powerOfTwo = 1;

	while (nodeIndex > maxLevelIndex)
	{
		powerOfTwo *= 2;
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


//Frontline 190314 - DUBuf IUBuf check function 완성하자

__global__ void check_delete_update_buffer()

bool ParallelHeapManager::is_delete_update_buffer_empty()
{
	int* result;
	cudaMalloc((void**)&result, sizeof(int));
}

bool ParallelHeapManager::is_insert_update_buffer_empty()
{

}



void ParallelHeapManager::print_heap()
{
	std::cout << "Print heap: " <<m_heapSize<<std::endl;
	for (int i=0; i<m_heapSize; i++)
	{
		std::cout << "Heap[" << i << "]: " << m_heap[i] << std::endl;
	}
	std::cout << std::endl << std::endl;
}