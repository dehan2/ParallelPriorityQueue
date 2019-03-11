#include "ParallelHeapManager.cuh"
#include "constForParallelHeap.h"
#include <thrust/sort.h>
#include <thrust/swap.h>




////////////////////////////////////////////////////////////////////
// TO DO 190306 - Refer this page, make it possible to use vector in kernel
// https://gist.github.com/docwhite/843f17e33e4c1f2b531a14a4bdfe90ec
////////////////////////////////////////////////////////////////////



void ParallelHeapManager::push(const double& entity)
{
	//It is guaranteed that there is no pop operation until every item is inserted
	//int destinationIndex = m_heap.size() + m_insertUpdateBuffer_oddLevel.size()+ m_insertUpdateBuffer_evenLevel.size();
	InsertItem item;
	item.currNode = 0;
	item.destination = m_heap.size();
	item.entity = entity;
	m_insertUpdateBuffer_evenLevel.push_back(item);
	m_heap.push_back(INT_MAX);
}



double ParallelHeapManager::pop()
{
	double topEntity = m_heap.front();
	
	m_deleteUpdateBuffer_evenLevel.push_back(0);
	m_heap.front() == m_heap.back();
	m_heap.pop_back();

	return topEntity;
}



void ParallelHeapManager::initiate_delete_update(const bool& isOdd)
{
	double* heapPtr = raw_pointer_cast(m_heap.data());
	int* tempBuffer_device, *tempBuffer_host;
	int bufferSize = 0;

	if (isOdd)
	{
		if (!m_deleteUpdateBuffer_oddLevel.empty())
		{
			bufferSize = m_deleteUpdateBuffer_oddLevel.size();
			int* deleteUpdateBufferPtr_odd = raw_pointer_cast(m_deleteUpdateBuffer_oddLevel.data());
			cudaMalloc((void**)&tempBuffer_device, sizeof(int)*bufferSize);

			delete_update <<<N, N >>>(heapPtr, m_heap.size(), 
				deleteUpdateBufferPtr_odd, bufferSize, tempBuffer_device);
			cudaDeviceSynchronize();

			m_deleteUpdateBuffer_oddLevel.clear();
		}
	}
	else
	{
		if (!m_deleteUpdateBuffer_evenLevel.empty())
		{
			bufferSize = m_deleteUpdateBuffer_evenLevel.size();
			int* deleteUpdateBufferPtr_even = raw_pointer_cast(m_deleteUpdateBuffer_evenLevel.data());
			cudaMalloc((void**)&tempBuffer_device, sizeof(int)*bufferSize);

			delete_update <<<N, N >>>(heapPtr, m_heap.size(),
				deleteUpdateBufferPtr_even, bufferSize, tempBuffer_device);
			cudaDeviceSynchronize();
			m_deleteUpdateBuffer_evenLevel.clear();
		}
	}

	if (bufferSize > 0)
	{
		tempBuffer_host = new int[bufferSize];
		cudaMemcpy(tempBuffer_device, tempBuffer_host, bufferSize * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(tempBuffer_device);

		transfer_delete_buffer_info(tempBuffer_host, bufferSize, isOdd);
		delete[] tempBuffer_host;
	}
}



void ParallelHeapManager::transfer_delete_buffer_info(int* tempBuffer_host, int bufferSize, bool isOdd)
{
	host_vector<int> bufferInfo;

	for (int i = 0; i < bufferSize; i++)
	{
		if (tempBuffer_host[i] != -1)
			bufferInfo.push_back(tempBuffer_host[i]);
	}

	if (isOdd)
	{
		copy(bufferInfo.begin(), bufferInfo.end(), m_deleteUpdateBuffer_evenLevel.end());
	}
	else
	{
		copy(bufferInfo.begin(), bufferInfo.end(), m_deleteUpdateBuffer_oddLevel.end());
	}
}



void ParallelHeapManager::initiate_insert_update(const bool& isOdd)
{
	double* heapPtr = raw_pointer_cast(m_heap.data());
	InsertItem* tempBuffer_device, *tempBuffer_host;
	int bufferSize = 0;

	if (isOdd)
	{
		if (!m_insertUpdateBuffer_oddLevel.empty())
		{
			bufferSize = m_insertUpdateBuffer_oddLevel.size();
			InsertItem* insertUpdateBufferPtr_odd = raw_pointer_cast(m_insertUpdateBuffer_oddLevel.data());
			cudaMalloc((void**)&tempBuffer_device, sizeof(InsertItem)*bufferSize);

			insert_update << <N, N >> > (heapPtr, m_heap.size(),
				insertUpdateBufferPtr_odd, bufferSize, tempBuffer_device);

			cudaDeviceSynchronize();
			m_insertUpdateBuffer_oddLevel.clear();
		}
	}
	else
	{
		if (!m_insertUpdateBuffer_evenLevel.empty())
		{
			bufferSize = m_insertUpdateBuffer_evenLevel.size();
			InsertItem* insertUpdateBufferPtr_even = raw_pointer_cast(m_insertUpdateBuffer_evenLevel.data());
			cudaMalloc((void**)&tempBuffer_device, sizeof(InsertItem)*bufferSize);

			insert_update << <N, N >> > (heapPtr, m_heap.size(),
				insertUpdateBufferPtr_even, bufferSize, tempBuffer_device);

			cudaDeviceSynchronize();
			m_insertUpdateBuffer_evenLevel.clear();
		}
	}

	if (bufferSize > 0)
	{
		tempBuffer_host = new InsertItem[bufferSize];
		cudaMemcpy(tempBuffer_device, tempBuffer_host, bufferSize * sizeof(InsertItem), cudaMemcpyDeviceToHost);
		cudaFree(tempBuffer_device);

		transfer_insert_buffer_info(tempBuffer_host, bufferSize, isOdd);
		delete[] tempBuffer_host;
	}
}



void ParallelHeapManager::transfer_insert_buffer_info(InsertItem* tempBuffer_host, int bufferSize, bool isOdd)
{
	host_vector<InsertItem> bufferInfo;

	for (int i = 0; i < bufferSize; i++)
	{
		if (tempBuffer_host[i].currNode != -1)
			bufferInfo.push_back(tempBuffer_host[i]);
	}

	if (isOdd)
	{
		copy(bufferInfo.begin(), bufferInfo.end(), m_insertUpdateBuffer_evenLevel.end());
	}
	else
	{
		copy(bufferInfo.begin(), bufferInfo.end(), m_insertUpdateBuffer_oddLevel.end());
	}
}



__global__ void delete_update(double* heapPtr, int heapSize,
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
					double* sortingBuffer;
					cudaMalloc((void**)&sortingBuffer, 3 * sizeof(double));
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

					cudaFree(sortingBuffer);
				}
				else
				{
					//Only left node exists.
					if (heapPtr[currIndex] > heapPtr[leftIndex])
					{
						double temp = heapPtr[currIndex];
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
		tid += N;
	}
}



__device__ void sort_three_entities(double* entities)
{
	if (entities[0] > entities[2])
	{
		double temp = entities[0];
		entities[0] = entities[2];
		entities[2] = temp;
	}

	if (entities[0] > entities[1])
	{
		double temp = entities[0];
		entities[0] = entities[1];
		entities[1] = temp;
	}
	
	if (entities[1] > entities[2])
	{
		double temp = entities[1];
		entities[1] = entities[2];
		entities[2] = temp;
	}
}



__global__ void insert_update(double* heapPtr, int heapSize,
	InsertItem* insertUpdateBufferPtr_current, int insertUpdateBufferSize
	, InsertItem* insertUpdateBufferPtr_next)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < insertUpdateBufferSize)
	{
		InsertItem currItem = insertUpdateBufferPtr_current[tid];

		if (currItem.currNode == currItem.destination)
		{
			//if (heapSize <= currItem.destination)
			//	heap.resize(currItem.destination + 1);

			heapPtr[currItem.destination] = currItem.entity;

			InsertItem dummyItem;
			dummyItem.destination = -1;
			insertUpdateBufferPtr_next[tid] = dummyItem;
		}
		else
		{
			double entityForNextLevel = currItem.entity;
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

			insertUpdateBufferPtr_next[tid] = nextItem;
		}
		tid += N;
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