#include <iostream>
#include "ParallelHeapManager.cuh"

#include <list>
#include <thrust/device_vector.h>

using namespace std;

int main()
{
	ParallelHeapManager heapManager;
	list<double> entities = { 100,101,102,103,104,105,106,107,108,109,110};

	for (auto& entity : entities)
		heapManager.push(entity);

	int index = 0;
	while (!heapManager.empty())
	{
		double popped = heapManager.pop();
		cout << "Heap[" << index << "]: " << popped << endl;
		index++;
	}
}