#ifndef VAST_MEMORY_H_
#define VAST_MEMORY_H_

#include <queue>
#include <mutex>

namespace otl {
    template <typename T>
    class vast_memory {
        public:
        vast_memory(int group_size, int groups){
            m_group_size = group_size;
            m_groups_num = groups;

            m_memory = new T[groups * group_size];
            for (int i = 0;i < groups; i++) {
                m_queue.push(i);
            }
        }

        ~vast_memory() {
            if (m_groups_num != m_queue.size()) {
                            std::cout << "Expect groups num:" << m_groups_num << ", but now is " 
                << m_queue.size() << std::endl;
                std::cout << "Something wrong." << std::endl; 
            }
            else {
                std::cout << "Vast memory is OK." << std::endl;
            }


            if(m_memory)
            delete[] m_memory;
        }

        T* get_memory(int& idx) {
            T* memory = nullptr;
            {
                std::unique_lock<std::mutex> locker(m_mutex);
                if (!m_queue.empty()) {
                    idx = m_queue.front();
                    memory = m_memory +  idx * m_group_size;
                    m_queue.pop();
                }
            }
            return memory;
        }

        void put_memory_back(int i) {
            {
                std::unique_lock<std::mutex> locker(m_mutex);
                if (m_queue.size() < m_groups_num) {
                    m_queue.push(i);
                }
            }
        }

        private:
        int m_groups_num;
        int m_group_size;
        std::mutex m_mutex;
        std::queue<int> m_queue;
        T* m_memory = nullptr;
    };
}


#endif