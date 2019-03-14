#pragma once

#include <mutex>
#include <list>
#include <future>
#include <thread>
#include <atomic>
#include "constForParallelHeap.h"
#include "ParallelHeapManager.h"

using namespace std;


class ParallelHeapController
{
	ParallelHeapManager m_parallelHeap;

	list<PPQ_COMMAND> m_commandQ;
	list<float> m_pushEntityQ;

	thread* m_thread = nullptr;
	mutex m_commandLock;
	mutex m_popLock;

	promise<float> m_minKey;
	atomic_bool m_commandProcessStatus = false;

public:
	ParallelHeapController() { initiate_loop(); }
	~ParallelHeapController() = default;

	void request_push(list<float>& entities);
	double request_pop();

	void initiate_loop();
	void kill_loop();
	void process_commands();
};

