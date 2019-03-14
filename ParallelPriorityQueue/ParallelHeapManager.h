#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <tuple>
#include "cuda.h"
#include "constForParallelHeap.h"

using namespace thrust;

struct InsertItem
{
	int currNode;
	int destination;
	float entity;
};

struct InsertItem2
{
	int destination;
	float entity;
};

class ParallelHeapManager
{
private:
	device_vector<float> m_heap;
	device_vector<bool> m_deleteUpdateBuffer;
	device_vector<InsertItem2> m_insertUpdateBuffer;
	int m_heapSize;
	int m_heapCapacity;


	
	/*device_vector<InsertItem> m_insertUpdateBuffer_oddLevel;
	device_vector<InsertItem> m_insertUpdateBuffer_evenLevel;
	device_vector<int> m_deleteUpdateBuffer_oddLevel;
	device_vector<int> m_deleteUpdateBuffer_evenLevel;*/


public:
	ParallelHeapManager();
	inline ~ParallelHeapManager() = default;

	inline bool empty() const { return (m_heapSize == 0); }
	void push(const float& entity);
	float pop();

	void increase_heap_size();
	void initialize_heap_and_buffer();

	/*inline bool isInsertUpdateBufferEmpty_odd() const { return m_insertUpdateBuffer_oddLevel.empty(); }
	inline bool isInsertUpdateBufferEmpty_even() const { return m_insertUpdateBuffer_evenLevel.empty(); }
	inline bool isDeleteUpdateEmpty_odd() const { return m_deleteUpdateBuffer_oddLevel.empty(); }
	inline bool isDeleteUpdateEmpty_even() const { return m_deleteUpdateBuffer_evenLevel.empty(); }*/

	void initiate_delete_update(const bool& isOdd);
	void initiate_insert_update(const bool& isOdd);

	bool is_delete_update_buffer_empty();
	bool is_insert_update_buffer_empty();

	void print_heap();
};

