#pragma once
#include <thrust/device_vector.h>
#include <vector>
#include <tuple>

using namespace thrust;

class ParallelHeapManager
{
private:
	device_vector<double> m_heap;
	std::vector<std::tuple<int, int, double>> m_insertUpdateBuffer;
	std::vector<std::pair<int, double>> m_deleteUpdateBuffer;

public:
	ParallelHeapManager() {}
	~ParallelHeapManager() {}

	bool empty() const { return m_heap.empty(); }
	void push(const double& entity);
	double pop();


	int reassign_items_for_current_and_child_nodes(int index);

	int calculate_node_level(int nodeIndex);
	bool is_index_bit_in_number_zero(int number, int index);
};


__global__ void initiate_delete_update(device_vector<double>& heap, bool* isGoingLeft, double* entity);

__device__ void insert_update();
__device__ void delete_update();
	
