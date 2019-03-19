#include "ParallelHeapController.h"

void ParallelHeapController::request_push(list<float>& entities)
{
	//cout << "Push requested" << endl;
	m_commandLock.lock();
	m_commandQ.push_back(PPQ_PUSH);
	m_pushEntityQ.insert(m_pushEntityQ.end(), entities.begin(), entities.end());
	m_commandLock.unlock();
}



double ParallelHeapController::request_pop()
{
	m_commandLock.lock();
	m_commandQ.push_back(PPQ_POP);
	m_commandLock.unlock();
	
	auto minKey_future = m_minKey.get_future();
	double popResult = minKey_future.get();
	m_minKey = promise<float>();

	return popResult;
}



void ParallelHeapController::initiate_loop()
{
	if (m_thread == nullptr)
		m_thread = new thread(&ParallelHeapController::process_commands, this);
}



void ParallelHeapController::kill_loop()
{
	if (m_thread == nullptr)
		return;

	{
		lock_guard<mutex> lock(m_commandLock);
		m_commandQ.push_back(PPQ_EXIT);
	}
	
	m_thread->join();
	delete m_thread;
	m_thread = nullptr;
}




void ParallelHeapController::process_commands()
{
	//cout << "Process command" << endl;
	while (true)
	{
		if (!m_commandQ.empty())
		{
			lock_guard<mutex> commandGuard(m_commandLock);

			PPQ_COMMAND command = m_commandQ.front();
			m_commandQ.pop_front();
			switch (command)
			{
			case PPQ_POP:
			{
#ifdef PRINT_PROGRESS
				cout << "Command: pop" << endl;
				m_parallelHeap.print_heap();
#endif

				if (m_parallelHeap.is_insert_update_buffer_empty())
				{
					//cout << "Insert buffer is empty" << endl;
					float popResult = m_parallelHeap.pop();
					m_minKey.set_value(popResult);
					m_commandQ.push_front(PPQ_UPDATE);
				}
				else
				{
					//cout << "Insert buffer is not empty" << endl;
					m_commandQ.push_back(PPQ_POP);
				}
				break;
			}
			case PPQ_PUSH:
			{
#ifdef PRINT_PROGRESS
				cout << "Command: push - " << m_pushEntityQ.front()<< endl;
				m_parallelHeap.print_heap();
#endif
				
				m_parallelHeap.push(m_pushEntityQ.front());
				m_pushEntityQ.pop_front();
				if (!m_pushEntityQ.empty())
					m_commandQ.push_front(PPQ_PUSH);

				m_commandQ.push_front(PPQ_UPDATE);
				break;
			}
			case PPQ_UPDATE:
			{
#ifdef PRINT_PROGRESS
				cout << "Command: update" << endl;
				m_parallelHeap.print_heap();
#endif
				
				m_parallelHeap.initiate_delete_update(false);
				m_parallelHeap.initiate_insert_update(false);
				m_parallelHeap.initiate_delete_update(true);
				m_parallelHeap.initiate_insert_update(true);

				//cout << "After update" << endl;
				//m_parallelHeap.print_heap();

				if (!m_parallelHeap.is_delete_update_buffer_empty()
					|| !m_parallelHeap.is_insert_update_buffer_empty())
					m_commandQ.push_back(PPQ_UPDATE);

				break;
			}
			case PPQ_EXIT:
			{
				if (m_commandQ.empty())
				{
					//cout << "Kill thread" << endl;
					return;
				}
				else
				{
					m_commandQ.push_back(PPQ_EXIT);
					break;
				}
			}
			default:
				break;
			}	//Switch end
		}
		else
		{
			this_thread::sleep_for(10ms);
		}
	}
}
