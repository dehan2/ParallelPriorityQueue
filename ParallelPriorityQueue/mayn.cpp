#include <iostream>
#include <list>
#include "ParallelHeapController.h"

#include <queue>
#include <chrono>
#include <random>

using namespace std;

int main()
{
	list<float> entities;
	random_device rd;  //Will be used to obtain a seed for the random number engine
	mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	uniform_real_distribution<> dis(1.0, 100.0);
	for (int i = 0; i < 1E3; i++)
	{
		entities.push_back(dis(gen));
	}



	//list<float> entities = { 13, 56, 38, 49, 57, 99, 37, 53, 55, 17, 97, 92, 74, 44, 23, 34, 15, 73, 1, 72 };
	
	// Parallel Priority Queue
	cout << "PPQ Computation Start" << endl;

	chrono::steady_clock::time_point begin = chrono::steady_clock::now();

	ParallelHeapController heapController;
	heapController.request_push(entities);

	list<float> results;
	for (int i = 0; i < entities.size(); i++)
	{
		float minKey = heapController.request_pop();
		results.push_back(minKey);	
	}
	heapController.kill_loop();

	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	auto PPQTime = chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1E6;

	/*int counter = 0;
	for (auto& result : results)
	{
		cout << "Heap[" << counter << "]: " << result <<endl;
		counter++;
	}*/

	cout << "Computation finish!" << endl;

	//std prioiry queue
	begin = chrono::steady_clock::now();

	priority_queue<float> PQ;
	for (auto& entity : entities)
	{
		PQ.push(entity);
	}

	list<float> result_PQ;
	while (!PQ.empty())
	{
		float minKey = PQ.top();
		PQ.pop();
		result_PQ.push_back(minKey);
	}

	end = chrono::steady_clock::now();
	auto SPQTime = chrono::duration_cast<chrono::nanoseconds>(end - begin).count()/ 1E6;

	cout << "Test result: PPQ - " << PPQTime << "ms, SPQ: " << SPQTime << "ms, ratio: " <<PPQTime/SPQTime<< endl;
}