#include <iostream>
#include "ParallelHeapManager.cuh"

#include <list>
#include "ParallelHeapController.h"

using namespace std;

int main()
{
	ParallelHeapController heapController;
	list<double> entities = { 13, 56, 38, 49, 57, 99, 37, 53, 55, 17, 97, 92, 74, 44, 23, 34, 15, 73, 1, 72 };
	heapController.request_push(entities);

	for (int i = 0; i < entities.size(); i++)
	{
		double minKey = heapController.request_pop();
		cout << "Heap[" << i << "]: " << minKey << endl;
	}
}