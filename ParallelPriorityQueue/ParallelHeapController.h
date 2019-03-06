#pragma once

#include <mutex>
#include <list>
#include <future>
#include <thread>
#include "constForParallelHeap.h"
#include "ParallelHeapManager.cuh"

using namespace std;


class ParallelHeapController
{
	ParallelHeapManager m_parallelHeap;

	list<PPQ_COMMAND> m_commandQ;
	list<double> m_pushEntityQ;
	mutex m_commandLock;
	mutex m_popLock;

	promise<double> m_minKey;
	promise<bool> m_commandProcessStatus;

public:
	ParallelHeapController() = default;
	~ParallelHeapController() = default;

	void request_push(list<double>& entities);
	double request_pop();

	void initiate_command_process_thread();
	void process_commands();
};

