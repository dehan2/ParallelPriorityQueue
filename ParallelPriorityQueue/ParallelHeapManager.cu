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
	int destinationIndex = m_heap.size() + m_insertUpdateBuffer_oddLevel.size()+ m_insertUpdateBuffer_evenLevel.size();
	InsertItem item;
	item.currNode = 0;
	item.destination = destinationIndex;
	item.entity = entity;
	m_insertUpdateBuffer_evenLevel.push_back(item);
}



double ParallelHeapManager::pop()
{
	double topEntity = m_heap.front();
	fetch_last_entity<<<1, 1 >>>(m_heap, m_deleteUpdateBuffer_evenLevel);
	cudaDeviceSynchronize();
	return topEntity;
}



void ParallelHeapManager::initiate_delete_update(const bool& isOdd)
{
	if (isOdd)
	{
		if (!m_deleteUpdateBuffer_oddLevel.empty())
		{
			delete_update <<<N, N >>> (m_heap, m_deleteUpdateBuffer_oddLevel, m_deleteUpdateBuffer_evenLevel);
			cudaDeviceSynchronize();
			m_deleteUpdateBuffer_oddLevel.clear();
		}
	}
	else
	{
		if (!m_deleteUpdateBuffer_evenLevel.empty())
		{
			delete_update <<<N, N >>> (m_heap, m_deleteUpdateBuffer_evenLevel, m_deleteUpdateBuffer_oddLevel);
			cudaDeviceSynchronize();
			m_deleteUpdateBuffer_evenLevel.clear();
		}
	}
}



void ParallelHeapManager::initiate_insert_update(const bool& isOdd)
{
	if (isOdd)
	{
		if (!m_insertUpdateBuffer_oddLevel.empty())
		{
			insert_update <<<N, N >>> (m_heap, m_insertUpdateBuffer_oddLevel, m_insertUpdateBuffer_evenLevel);
			cudaDeviceSynchronize();
			m_insertUpdateBuffer_oddLevel.clear();
		}
	}
	else
	{
		if (!m_insertUpdateBuffer_evenLevel.empty())
		{
			insert_update << <N, N >> > (m_heap, m_insertUpdateBuffer_evenLevel, m_insertUpdateBuffer_oddLevel);
			cudaDeviceSynchronize();
			m_insertUpdateBuffer_evenLevel.clear();
		}
	}
}



__global__ void fetch_last_entity(device_vector<double>& heap, device_vector<int>& deleteUpdateBuffer_evenLevel)
{
	if (heap.size() > 1)
	{
		deleteUpdateBuffer_evenLevel.push_back(0);
		heap.front() == heap.back();
		heap.pop_back();
	}
	else
	{
		//Do nothing
	}
}



__global__ void delete_update(device_vector<double>& heap, device_vector<int>& deleteUpdateBuffer_currentLevel, device_vector<int>& deleteUpdateBuffer_nextLevel)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < deleteUpdateBuffer_currentLevel.size())
	{
		int currIndex = deleteUpdateBuffer_currentLevel[tid];
		if (currIndex < heap.size())
		{
			int leftIndex = 2 * currIndex + 1;
			int rightIndex = 2 * currIndex + 2;
			if (leftIndex < heap.size())
			{
				if (rightIndex < heap.size())
				{
					//All three nodes exist.
					device_vector<double> buffer;
					buffer.push_back(heap[currIndex]);
					buffer.push_back(heap[leftIndex]);
					buffer.push_back(heap[rightIndex]);

					sort(buffer.begin(), buffer.end());

					heap[currIndex] = buffer[0];
					if (heap[leftIndex] < heap[rightIndex])
					{
						heap[leftIndex] = buffer[2];
						heap[rightIndex] = buffer[1];
						deleteUpdateBuffer_nextLevel.push_back(leftIndex);
					}
					else
					{
						heap[leftIndex] = buffer[1];
						heap[rightIndex] = buffer[2];
						deleteUpdateBuffer_nextLevel.push_back(rightIndex);
					}
				}
				else
				{
					//Only left node exists.
					if (heap[currIndex] > heap[leftIndex])
					{
						swap(heap[currIndex], heap[leftIndex]);
					}
				}
			}
			else
			{
				//Only current node exists.
				//Do nothing
			}
		}
		tid += N;
	}
}



__global__ void insert_update(device_vector<double>& heap, 
	device_vector<InsertItem>& insertUpdateBuffer_currentLevel,
	device_vector<InsertItem>& insertUpdateBuffer_nextLevel)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < insertUpdateBuffer_currentLevel.size())
	{
		InsertItem currItem = insertUpdateBuffer_currentLevel[tid];

		if (currItem.currNode == currItem.destination)
		{
			if (heap.size() <= currItem.destination)
				heap.resize(currItem.destination + 1);

			heap[currItem.destination] = currItem.entity;
		}
		else
		{
			double entityForNextLevel = currItem.entity;
			if (heap[currItem.currNode] > entityForNextLevel)
			{
				entityForNextLevel = heap[currItem.currNode];
				heap[currItem.currNode] = currItem.entity;
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

			insertUpdateBuffer_nextLevel.push_back(nextItem);
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