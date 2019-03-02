#include "ParallelHeapManager.cuh"
#include <thrust/sort.h>
#include <thrust/swap.h>


void ParallelHeapManager::push(const double& entity)
{
	int targetNodeIndex = m_heap.size() + m_insertUpdateBuffer.size();
	m_insertUpdateBuffer.push_back(std::make_tuple(0, targetNodeIndex, entity));
}



double ParallelHeapManager::pop()
{
	double topEntity = m_heap.front();
	bool* isGoingLeft_device;
	double* entity_device;
	cudaMalloc((void**)&isGoingLeft_device, sizeof(bool));
	cudaMalloc((void**)&entity_device, sizeof(double));
	initiate_delete_update <<<1, 1 >>> (isGoingLeft, entity);

	bool isGoingLeft;
	double entity;
	cudaMemcpy(&isGoingLeft, isGoingLeft_device, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(&entity, entity_device, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(isGoingLeft_device);
	cudaFree(entity_device);

	if (isGoingLeft)
		m_deleteUpdateBuffer.push_back(std::make_pair(1, entity));
	else
		m_deleteUpdateBuffer.push_back(std::make_pair(2, entity));

	return topEntity;
}








int ParallelHeapManager::reassign_items_for_current_and_child_nodes(int index)
{
	if ((2 * index + 1) < m_heap.size() && (2 * index + 2) < m_heap.size())
	{
		//Current node have both child

		device_vector<double> buffer;
		buffer.push_back(m_heap[index]);
		buffer.push_back(m_heap[2 * index + 1]);
		buffer.push_back(m_heap[2 * index + 2]);

		sort(buffer.begin(), buffer.end());

		int reassignResult = -1;		//		-1: heap property satisfied /  else: index for next update node
		int newIndex = 0;

		m_heap[index] = buffer[0];
		if (m_heap[2 * index + 1] < m_heap[2 * index + 2])
		{
			m_heap[2 * index + 1] = buffer[2];
			m_heap[2 * index + 2] = buffer[1];
			newIndex = 2 * index + 1;
		}
		else
		{
			m_heap[2 * index + 1] = buffer[1];
			m_heap[2 * index + 2] = buffer[2];
			newIndex = 2 * index + 2;
		}

		//check heap property
		if (m_heap[newIndex] <= m_heap[2 * newIndex + 1]
			&& m_heap[newIndex] <= m_heap[2 * newIndex + 2])
		{
			reassignResult = -1;
		}
		else
		{
			reassignResult = newIndex;
		}

		return reassignResult;
	}
	else if ((2 * index + 1) < m_heap.size())
	{
		//Current node have left child only
		if (m_heap[index] > m_heap[2 * index + 1])
			swap(m_heap[index], m_heap[2 * index + 1]);
		
		return -1;
	}
	else
	{
		//Current node have no child
		return -1;
	}
}



int ParallelHeapManager::calculate_node_level(int nodeIndex)
{
	int level = 1;
	int maxLevelIndex = 0;

	while (nodeIndex > maxLevelIndex)
	{
		maxLevelIndex += pow(2, level);
		level++;
	}

	return level;
}



bool ParallelHeapManager::is_index_bit_in_number_zero(int number, int index)
{
	int translatedNum = number >> (index - 1);
	int result = translatedNum % 2;
	return result == 0;
}

//Frontline 190302 - Äí´Ù¾î·Á¿ý
__global__ void initiate_delete_update(device_vector<double>& heap, int* updateResult, double* entity)
{
	device_vector<double>::iterator itForLastElement = --heap.end();
	heap.front() = *itForLastElement;
	heap.erase(itForLastElement);

	device_vector<double> sortingBuffer;
	sortingBuffer.push_back(heap[0]);
	sortingBuffer.push_back(heap[1]);
	sortingBuffer.push_back(heap[2]);

	sort(sortingBuffer.begin(), sortingBuffer.end());
	heap.front() = sortingBuffer[0];
	if ((2 * index + 1) < m_heap.size() && (2 * index + 2) < m_heap.size())
	{
		//Current node have both child

		device_vector<double> buffer;
		buffer.push_back(m_heap[index]);
		buffer.push_back(m_heap[2 * index + 1]);
		buffer.push_back(m_heap[2 * index + 2]);

		sort(buffer.begin(), buffer.end());

		int reassignResult = -1;		//		-1: heap property satisfied /  else: index for next update node
		int newIndex = 0;

		m_heap[index] = buffer[0];
		if (m_heap[2 * index + 1] < m_heap[2 * index + 2])
		{
			m_heap[2 * index + 1] = buffer[2];
			m_heap[2 * index + 2] = buffer[1];
			newIndex = 2 * index + 1;
		}
		else
		{
			m_heap[2 * index + 1] = buffer[1];
			m_heap[2 * index + 2] = buffer[2];
			newIndex = 2 * index + 2;
		}

		//check heap property
		if (m_heap[newIndex] <= m_heap[2 * newIndex + 1]
			&& m_heap[newIndex] <= m_heap[2 * newIndex + 2])
		{
			reassignResult = -1;
		}
		else
		{
			reassignResult = newIndex;
		}

		return reassignResult;
	}
	else if ((2 * index + 1) < m_heap.size())
	{
		//Current node have left child only
		if (m_heap[index] > m_heap[2 * index + 1])
			swap(m_heap[index], m_heap[2 * index + 1]);

		return -1;
	}
	else
	{
		//Current node have no child
		return -1;
	}


	m_nodesForDeleteUpdate.push_back(0);
	while (!m_nodesForDeleteUpdate.empty())
	{
		int reassignResult = reassign_items_for_current_and_child_nodes(m_nodesForDeleteUpdate.front());
		m_nodesForDeleteUpdate.erase(m_nodesForDeleteUpdate.begin());
		if (reassignResult != -1)
			m_nodesForDeleteUpdate.push_back(reassignResult);
	}

	return minKey;
}




















__global__ void ParallelHeap::push(device_vector<double>& heap, const double& entity)
{
	ParallelHeap::insert_update<<<1, 1 >>> (entity);


	device_vector<int> nodesForInsertUpdate;
	__device__ double 


	m_entitiesToInsert.push_back(entity);
	m_nodesForInsertUpdate.push_back(0);
	int targetNodeIndex = m_heap.size();
	int levelOfTargetNode = calculate_node_level(targetNodeIndex);

	while (!m_nodesForInsertUpdate.empty())
	{
		int currNodeIndex = m_nodesForInsertUpdate.front();
		if (targetNodeIndex == currNodeIndex)
		{
			m_heap.push_back(m_entitiesToInsert.front());
		}
		else
		{
			m_entitiesToInsert.push_back(m_heap[currNodeIndex]);
			sort(m_entitiesToInsert.begin(), m_entitiesToInsert.end());
			m_heap[currNodeIndex] = m_entitiesToInsert.front();
			int levelOfCurrNode = calculate_node_level(currNodeIndex);
			bool goLeft = is_index_bit_in_number_zero(targetNodeIndex + 1, (levelOfTargetNode - levelOfCurrNode));
			if (goLeft)
				m_nodesForInsertUpdate.push_back(2 * currNodeIndex + 1);
			else
				m_nodesForInsertUpdate.push_back(2 * currNodeIndex + 2);
		}
		m_entitiesToInsert.erase(m_entitiesToInsert.begin());
		m_nodesForInsertUpdate.erase(m_nodesForInsertUpdate.begin());
	}
}



__global__ void ParallelHeap::pop(device_vector<double>& heap, double* result)
{
	double minKey = m_heap.front();
	device_vector<double>::iterator itForLastElement = --m_heap.end();
	m_heap.front() = *itForLastElement;
	m_heap.erase(itForLastElement);
	m_nodesForDeleteUpdate.push_back(0);
	while (!m_nodesForDeleteUpdate.empty())
	{
		int reassignResult = reassign_items_for_current_and_child_nodes(m_nodesForDeleteUpdate.front());
		m_nodesForDeleteUpdate.erase(m_nodesForDeleteUpdate.begin());
		if (reassignResult != -1)
			m_nodesForDeleteUpdate.push_back(reassignResult);
	}

	return minKey;
}