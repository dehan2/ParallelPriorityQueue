#pragma once
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
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
	void initiate_insert_update(const bool& isOdd);
};

__global__ void fetch_last_entity(device_vector<double>& heap, device_vector<int>& deleteUpdateBuffer_evenLevel);
__global__ void delete_update(device_vector<double>& heap, device_vector<int>& deleteUpdateBuffer_currentLevel, device_vector<int>& deleteUpdateBuffer_nextLevel);
__global__ void insert_update(device_vector<double>& heap, device_vector<InsertItem>& insertUpdateBuffer_currentLevel,
														device_vector<InsertItem>& insertUpdateBuffer_nextLevel);

__device__ int calculate_node_level(int nodeIndex);
__device__ bool is_index_bit_in_number_zero(int number, int index);

__device__ void insert_update();
__device__ void delete_update();
	
