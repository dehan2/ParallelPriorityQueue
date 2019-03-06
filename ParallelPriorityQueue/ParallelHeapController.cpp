#include "ParallelHeapController.h"

void ParallelHeapController::request_push(list<double>& entities)
{
	m_commandLock.lock();
	m_commandQ.push_back(PPQ_PUSH);
	m_pushEntityQ.insert(m_pushEntityQ.end(), entities.begin(), entities.end());
	initiate_command_process_thread();
	m_commandLock.unlock();
}



double ParallelHeapController::request_pop()
{
	m_commandLock.lock();
	m_commandQ.push_back(PPQ_POP);
	initiate_command_process_thread();
	m_commandLock.unlock();

	auto minKey_future = m_minKey.get_future();
	double popResult = minKey_future.get();
	m_minKey = promise<double>();

	return popResult;
}



void ParallelHeapController::initiate_command_process_thread()
{
	//check thread status
	auto commandProcessStatusFuture = m_commandProcessStatus.get_future();
	auto status = commandProcessStatusFuture.wait_for(0ms);
	if (status == future_status::ready)
	{
		thread([&]() {process_commands(); });
		m_commandProcessStatus = promise<bool>();
	}
}



void ParallelHeapController::process_commands()
{
	while (!m_commandQ.empty())
	{
		lock_guard<mutex> commandGuard(m_commandLock);

		PPQ_COMMAND command = m_commandQ.front();
		m_commandQ.pop_front();
		switch (command)
		{
		case PPQ_POP:
		{
			if (m_parallelHeap.isInsertUpdateBufferEmpty_odd() && m_parallelHeap.isInsertUpdateBufferEmpty_even())
			{
				double popResult = m_parallelHeap.pop();
				m_minKey.set_value(popResult);
				m_commandQ.push_front(PPQ_UPDATE);
			}
			else
				m_commandQ.push_back(PPQ_POP);
			break;
		}
		case PPQ_PUSH:
		{
			m_parallelHeap.push(m_pushEntityQ.front());
			m_pushEntityQ.pop_front();
			if (!m_pushEntityQ.empty())
				m_commandQ.push_front(PPQ_PUSH);

			m_commandQ.push_front(PPQ_UPDATE);	
			break;
		}
		case PPQ_UPDATE:
		{
			m_parallelHeap.initiate_delete_update(false);
			m_parallelHeap.initiate_insert_update(false);
			m_parallelHeap.initiate_delete_update(true);
			m_parallelHeap.initiate_insert_update(true);

			if (!m_parallelHeap.isDeleteUpdateEmpty_even()
				|| !m_parallelHeap.isInsertUpdateBufferEmpty_even())
				m_commandQ.push_back(PPQ_UPDATE);

			break;
		}
		}	//Switch end
	}

	m_commandProcessStatus.set_value(true);
}
