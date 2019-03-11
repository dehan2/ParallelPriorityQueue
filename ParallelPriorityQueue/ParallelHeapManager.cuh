#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <tuple>

using namespace thrust;

struct InsertItem
{
	int currNode;
	int destination;
	double entity;
};

class ParallelHeapManager
{
private:
	device_vector<double> m_heap;
	
	device_vector<InsertItem> m_insertUpdateBuffer_oddLevel;
	device_vector<InsertItem> m_insertUpdateBuffer_evenLevel;
	device_vector<int> m_deleteUpdateBuffer_oddLevel;
	device_vector<int> m_deleteUpdateBuffer_evenLevel;


public:
	inline ParallelHeapManager() = default;
	inline ~ParallelHeapManager() = default;

	inline bool empty() const { return m_heap.empty(); }
	void push(const double& entity);
	double pop();

	inline bool isInsertUpdateBufferEmpty_odd() const { return m_insertUpdateBuffer_oddLevel.empty(); }
	inline bool isInsertUpdateBufferEmpty_even() const { return m_insertUpdateBuffer_evenLevel.empty(); }
	inline bool isDeleteUpdateEmpty_odd() const { return m_deleteUpdateBuffer_oddLevel.empty(); }
	inline bool isDeleteUpdateEmpty_even() const { return m_deleteUpdateBuffer_evenLevel.empty(); }

	void initiate_delete_update(const bool& isOdd);
	void transfer_delete_buffer_info(int* tempBuffer_host, int bufferSize, bool isOdd);

	void initiate_insert_update(const bool& isOdd);
	void transfer_insert_buffer_info(InsertItem* tempBuffer_host, int bufferSize, bool isOdd);
};

__global__ void delete_update(double* heapPtr, int heapSize, 
	int* deleteUpdateBufferPtr_current, int bufferSize
	, int* deleteUpdateBufferPtr_next);

__device__ void sort_three_entities(double* entities);


__global__ void insert_update(double* heapPtr, int heapSize,
	InsertItem* insertUpdateBufferPtr_current, int bufferSize
	, InsertItem* insertUpdateBufferPtr_next);

__device__ int calculate_node_level(int nodeIndex);
__device__ bool is_index_bit_in_number_zero(int number, int index);
