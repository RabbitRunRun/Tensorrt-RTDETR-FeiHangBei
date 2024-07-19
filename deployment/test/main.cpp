#include <iostream>
#include "rtdetr.h"
#include "config.h"
#include <chrono>
#include <fstream>
#include "rtdetr_utils.h"
#include "otl/thread/thread_pool.h"

struct RedetrDeleter
{
    void operator()(seeta::Rtdetr* obj) const
    {
        if (obj)
        delete obj;
    }
};

int main_image_test(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: main image_path.\n");
        exit(-1);
    }

    const char* image_path = argv[1];
    cv::Mat image = cv::imread(image_path);
    // printf("image width:%d height:%d channels:%d", image.cols, image.rows, image.channels());

    Config config =  ReadConfig("config.ini");
    std:: cout << config << std::endl;

    std::unique_ptr<seeta::Rtdetr, RedetrDeleter> rtdetr(
        new seeta::Rtdetr(config.model.detector_model.c_str(), 
        config.parameter.detector_thresh));
    
    detect_result_group result_group;
    int test_count = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_count; ++i) 
    {
        result_group = rtdetr->detect(image.data, image.cols, image.rows, false);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "loop " << test_count << " times" << ", each processing spent " 
            << duration.count() * 1.0 / test_count << "ms" << std::endl; 
    std::cout << "results size: " << result_group.size << std::endl;
    for (int i = 0; i < result_group.size; ++i) {
        float score = result_group.data[i].score;
        bbox box = result_group.data[i].box;
        std::cout << i << " " << score << " " << box.x << " " << box.y 
                << " " << box.x + box.width << " " << box.y 
                << " " << box.x + box.width << " " << box.y + box.height 
                << " " << box.x << " "<< box.y+box.height
                <<" " << box.x + box.width / 2.0 << " " << box.y + box.height / 2.0 << std::endl;

        cv::rectangle(image, cv::Rect(box.x, box.y, box.width, box.height), CV_RGB(255, 0, 0), 3);
        cv::putText(image, std::to_string(score), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
    }

    printf("output image name:result.jpg\n");
    cv::imwrite("result.jpg", image);

    return 0;
}

int main_images_test(int argc, char** argv) {

    auto start = std::chrono::high_resolution_clock::now();
    Config config =  ReadConfig("config.ini");
    std::cout << config << std::endl;

    std::string images_path = config.parameter.image_path;
    std::string saved_path = config.parameter.save_path;
    if (!seeta::directory_exists(saved_path)) {
        std::cout << "Creating directory " << saved_path << std::endl;
        seeta::create_directory(saved_path);
    }

    std::vector<std::string> images = seeta::FindFilesRecursively(images_path,-1);
    std::cout << "Found " << images.size() << " images." << std::endl;

    std::unique_ptr<seeta::Rtdetr, RedetrDeleter> rtdetr(
        new seeta::Rtdetr(config.model.detector_model.c_str(), 
        config.parameter.detector_thresh));

    int images_size = images.size();
    for(int i = 0; i < images_size; ++i) {
        if (i % 200 == 0) {
            printf("Process:%d/%d\r", i+1, images_size);
            fflush(stdout);
        }

        std::string image_path = images_path + seeta::FileSeparator() + images[i];
        cv::Mat image;
        {
            // auto start = std::chrono::high_resolution_clock::now();
            image = cv::imread(image_path);
            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> duration = end - start;
            // std::cout << "Reading image spent " << duration.count() << "ms" << std::endl; 
        }


        detect_result_group result_group;
        result_group = rtdetr->detect(image.data, image.cols, image.rows, false);

        // write results to save path
        {
            // auto start = std::chrono::high_resolution_clock::now();

            std::string file_name = seeta::getFileName(images[i]);
            std::string base_name = seeta::getBaseName(file_name);
            std::string saved_txt = saved_path + "/" + base_name + ".txt";
            std::ofstream out(saved_txt);
            for (int j = 0; j < result_group.size; ++j) {
                float score = result_group.data[j].score;
                bbox box = result_group.data[j].box;
                int cls = result_group.data[j].cls;
                if (j != 0) out << std::endl;
                out << j + 1 << " " << cls << " " << score 
                    << " " << box.x << " " << box.y
                    << " " << box.x + box.width << " " << box.y
                    << " " << box.x + box.width << " " << box.y + box.height
                    << " " << box.x << " " << box.y + box.height
                    << " " << box.x + box.width / 2.0 << " " << box.y + box.height / 2.0;
            }
            out.close();
            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> duration = end - start;
            // std::cout << "Writing results to file spent " << duration.count() << "ms" << std::endl; 
        }

    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Processing " << images_size << " images spent " 
            << duration.count() * 1.0 << "ms" << std::endl; 

    return 0;
}

int main_images_multi_threads(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    Config config =  ReadConfig("config.ini");
    std::cout << config << std::endl;

    std::string images_path = config.parameter.image_path;
    std::string saved_path = config.parameter.save_path;
    if (!seeta::directory_exists(saved_path)) {
        std::cout << "Creating directory " << saved_path << std::endl;
        seeta::create_directory(saved_path);
    }

    std::vector<std::string> images = seeta::FindFilesRecursively(images_path,-1);
    std::cout << "Found " << images.size() << " images." << std::endl;

    // std::vector<std::unique_ptr<seeta::Rtdetr, RedetrDeleter>> rtdetrs(config.parameter.workers_num);
    // for (int num = 0; num < config.parameter.workers_num; ++num) {
    //     rtdetrs[num].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
    //                                     config.parameter.detector_thresh));
    // }

    // otl::ThreadPool thread_pool(config.parameter.workers_num);


    // init using thread pool
    std::vector<std::unique_ptr<seeta::Rtdetr, RedetrDeleter>> rtdetrs(config.parameter.workers_num);
    otl::ThreadPool thread_pool(config.parameter.workers_num);
    for (int i = 0; i < config.parameter.workers_num; i++) {
        thread_pool.run([&rtdetrs, &config](int idx){
        rtdetrs[idx].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
                                        config.parameter.detector_thresh));
        });
    }

    thread_pool.join();

    int images_size = images.size();
    for(int i = 0; i < images_size; ++i) {
        if (i % 200 == 0) {
            printf("Process:%d/%d\r", i+1, images_size);
            fflush(stdout);
        }
        thread_pool.run([&rtdetrs, &images, i, &images_path, &saved_path](int idx){
            // std::cout << "worker idx: " << idx << std::endl;
            std::string image_path = images_path + seeta::FileSeparator() + images[i];
            cv::Mat image = cv::imread(image_path);
            detect_result_group result_group;
            result_group = rtdetrs[idx]->detect(image.data, image.cols, image.rows, false);

            // write results to save path
            std::string file_name = seeta::getFileName(images[i]);
            std::string base_name = seeta::getBaseName(file_name);
            std::string saved_txt = saved_path + "/" + base_name + ".txt";
            std::ofstream out(saved_txt);
            for (int j = 0; j < result_group.size; ++j) {
                float score = result_group.data[j].score;
                bbox box = result_group.data[j].box;
                int cls = result_group.data[j].cls;
                if (j != 0) out << std::endl;
                out << j + 1 << " " << cls << " " << score 
                    << " " << box.x << " " << box.y
                    << " " << box.x + box.width << " " << box.y
                    << " " << box.x + box.width << " " << box.y + box.height
                    << " " << box.x << " " << box.y + box.height
                    << " " << box.x + box.width / 2.0 << " " << box.y + box.height / 2.0;
            }
            out.close();
        });
    }
    thread_pool.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Processing " << images_size << " images spent " 
            << duration.count() * 1.0 << "ms" << std::endl; 

    return 0;

}

#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "vast_memory.h"

struct InputInfo {
    std::shared_ptr<float> chw_data;
    std::string image;
    int origin_image_width;
    int origin_image_height;
};

struct InputInfoV2 {
    float* chw_data;
    int data_idx;

    std::string image;
    int origin_image_width;
    int origin_image_height;
};

struct InferResult {
    std::vector<detect_result> results;
    std::string image; // for txt file
};

static std::queue<InputInfo> inputQueue; // model input data buffer queue, including data and image file name
static std::queue<InputInfoV2> inputQueueV2; // model input data buffer queue, including data and image file name
static std::queue<InferResult> resultQueue; // results
static std::mutex inputMutex, resultMutex;
static std::condition_variable inputCondVar, resultCondVar;
static std::atomic<bool> preprocess_done(false);
static std::atomic<bool> infer_done(false);

static void preprocess_func(const std::string& images_path, const std::vector<std::string>& images,
                            const Config& config, int input_size) {
    int image_size = images.size();
	for (int i = 0; i < image_size; ++i) {
        // progress bar
        if (i % 200 == 0) {
            printf("Process:%d/%d\r", i+1, image_size);
            fflush(stdout);
        }

		int queue_size = 0;
        std::string image_path = images_path + seeta::FileSeparator() + images[i];
        cv::Mat image = cv::imread(image_path);

        InputInfo input_info;
        input_info.image = images[i];
        input_info.origin_image_width = image.cols;
        input_info.origin_image_height = image.rows;
        input_info.chw_data.reset(new float[1 * 3 * input_size * input_size], std::default_delete<float[]>());

        float scale_x,scale_y;
        int padding_top, padding_bottom, padding_left, padding_right;
        seeta::preprocess(image, input_size, input_size,
                    scale_x, scale_y, padding_top, padding_bottom, padding_left, padding_right, 
                    true, (float*)input_info.chw_data.get());
		{
			std::lock_guard<std::mutex> lock(inputMutex);
			inputQueue.push(input_info);
            // std::cout << "input image:" << input_info.image << std::endl;
			queue_size = inputQueue.size();
		}
        // notify one
		inputCondVar.notify_one();

        // buffered data is enough, just sleep a little
        if (queue_size >= 4 * config.parameter.workers_num)
           {
            // std::cout << "i am sleeping..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
           }
	}
    // done and notify all
    std::cout << "Preprocess_func finished!" << std::endl;
	preprocess_done = true;
	inputCondVar.notify_all();
}


static void preprocess_func_with_vast_memory(const std::string& images_path, const std::vector<std::string>& images,
                            const Config& config, int input_size, otl::vast_memory<float>& vast_memory) {
    int image_size = images.size();
	for (int i = 0; i < image_size; ++i) {
        // progress bar
        if (i % 200 == 0) {
            printf("Process:%d/%d\r", i+1, image_size);
            fflush(stdout);
        }

		int queue_size = 0;
        std::string image_path = images_path + seeta::FileSeparator() + images[i];
        cv::Mat image = cv::imread(image_path);

        InputInfoV2 input_info;
        input_info.image = images[i];
        input_info.origin_image_width = image.cols;
        input_info.origin_image_height = image.rows;

        while(true) {
            int idx;
            float* memory = vast_memory.get_memory(idx);
            // buffered data is not enough, just sleep a little
            if (memory == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            else {
                // got valid buffer from vast memory
                input_info.chw_data = memory;
                input_info.data_idx = idx;
                break;
            }
        }

        float scale_x,scale_y;
        int padding_top, padding_bottom, padding_left, padding_right;
        seeta::preprocess(image, input_size, input_size,
                    scale_x, scale_y, padding_top, padding_bottom, padding_left, padding_right, 
                    true, (float*)input_info.chw_data);
		{
			std::lock_guard<std::mutex> lock(inputMutex);
			inputQueueV2.push(input_info);
            // std::cout << "input image:" << input_info.image << std::endl;
			queue_size = inputQueueV2.size();
		}
        // notify one
		inputCondVar.notify_one();
	}
    // done and notify all
    std::cout << "Preprocess_func finished!" << std::endl;
	preprocess_done = true;
	inputCondVar.notify_all();
}

static void infer_func(std::vector<std::unique_ptr<seeta::Rtdetr, RedetrDeleter>>& rtdetrs,
        otl::ThreadPool& thread_pool, const Config& config) {
	while (true) {
        InputInfo info;
        info.chw_data = nullptr;
        {
            std::unique_lock<std::mutex> lock(inputMutex);
            inputCondVar.wait(lock, [] { return !inputQueue.empty() || preprocess_done; });

            if (inputQueue.size() <= 0 && preprocess_done) {
                std::cout << "Infer func finished!" << std::endl;
                break;
		    }
            if (inputQueue.size() >= 1) {
                info = inputQueue.front();
                inputQueue.pop();
            }
        }

		if (info.chw_data != nullptr) {
            std::shared_ptr<float> chw_data = info.chw_data;
            std::string image = info.image;
            // std::cout << "image: " << image << std::endl;
            int image_width = info.origin_image_width;
            int image_height = info.origin_image_height;
             
            // to multi threads inference
            thread_pool.run([&rtdetrs, chw_data, image, image_width, image_height](int idx) {
                    // std::cout << "into run" << std::endl;
                    // std::cout << "index: " << idx << ", ptr: " << rtdetrs[idx].get() << std::endl;
                    InferResult infer_result;
                    infer_result.results = rtdetrs[idx]->detect(chw_data.get(), image_width, image_height);
                    // std::cout << "after detect"<<std::endl;
                    infer_result.image = image;
                    
                    {
                        // put infer result to queue
                        std::lock_guard<std::mutex> resultLock(resultMutex);
                        // std::cout << "Got " << infer_result.image << " results into result queue." << std::endl;
                        resultQueue.push(std::move(infer_result));
                    }
                    // notify one
                    resultCondVar.notify_one();
            });   
		}

	}

    // join multi threads work
    thread_pool.join();

    infer_done = true;

    // notify all
	resultCondVar.notify_all();
}

static void infer_func_with_vast_memory(std::vector<std::unique_ptr<seeta::Rtdetr, RedetrDeleter>>& rtdetrs,
        otl::ThreadPool& thread_pool, const Config& config, otl::vast_memory<float>& vast_memory) {
	while (true) {
        InputInfoV2 info;
        info.chw_data = nullptr;
        {
            std::unique_lock<std::mutex> lock(inputMutex);
            inputCondVar.wait(lock, [] { return !inputQueueV2.empty() || preprocess_done; });

            if (inputQueueV2.size() <= 0 && preprocess_done) {
                std::cout << "Infer func finished!" << std::endl;
                break;
		    }
            if (inputQueueV2.size() >= 1) {
                info = inputQueueV2.front();
                inputQueueV2.pop();
            }
        }

		if (info.chw_data != nullptr) {
            float* chw_data = info.chw_data;
            int data_idx = info.data_idx;

            std::string image = info.image;
            // std::cout << "image: " << image << std::endl;
            int image_width = info.origin_image_width;
            int image_height = info.origin_image_height;
             
            // to multi threads inference
            thread_pool.run([&rtdetrs, chw_data, image, image_width, image_height, data_idx, &vast_memory](int idx) {
                    // std::cout << "into run" << std::endl;
                    // std::cout << "index: " << idx << ", ptr: " << rtdetrs[idx].get() << std::endl;
                    InferResult infer_result;
                    infer_result.results = rtdetrs[idx]->detect(chw_data, image_width, image_height);
                    // std::cout << "after detect"<<std::endl;

                    // put back memory to vast memory
                    vast_memory.put_memory_back(data_idx);

                    infer_result.image = image;
                    
                    {
                        // put infer result to queue
                        std::lock_guard<std::mutex> resultLock(resultMutex);
                        // std::cout << "Got " << infer_result.image << " results into result queue." << std::endl;
                        resultQueue.push(std::move(infer_result));
                    }
                    // notify one
                    resultCondVar.notify_one();
            });   
		}

	}

    // join multi threads work
    thread_pool.join();

    infer_done = true;

    // notify all
	resultCondVar.notify_all();
}

static void write_results_func(const std::string& saved_path) {
	while (true) {
        std::vector<InferResult> results;

        {
            std::unique_lock<std::mutex> lock(resultMutex);
            resultCondVar.wait(lock, [] { return infer_done || !resultQueue.empty() || (preprocess_done && inputQueue.empty()); });

            if (resultQueue.empty() && infer_done && inputQueue.empty() && preprocess_done) {
                std::cout << "Write results func finished!" << std::endl; 
                break;
            }
            // collect all results
            results.reserve(resultQueue.size());
            while  (!resultQueue.empty()) {
                InferResult& infer_result = resultQueue.front();
                results.emplace_back(std::move(infer_result));
                resultQueue.pop();
            }
        }
        
        // std::cout << "result size: " << results.size() << std::endl;
        for (int i = 0; i < results.size(); ++i) {
            InferResult& infer_result = results[i];
            // write results to save path
                std::string file_name = seeta::getFileName(infer_result.image);
                std::string base_name = seeta::getBaseName(file_name);
                std::string saved_txt = saved_path + "/" + base_name + ".txt";
                std::ofstream out(saved_txt);
                for (int j = 0; j < infer_result.results.size(); ++j) {
                    float score = infer_result.results[j].score;
                    bbox box = infer_result.results[j].box;
                    int cls = infer_result.results[j].cls;
                    if (j != 0) out << std::endl;
                    out << j + 1 << " " << cls << " " << score 
                        << " " << box.x << " " << box.y
                        << " " << box.x + box.width << " " << box.y
                        << " " << box.x + box.width << " " << box.y + box.height
                        << " " << box.x << " " << box.y + box.height
                        << " " << box.x + box.width / 2.0 << " " << box.y + box.height / 2.0;
                }
                out.close();
        }

	}
}


static void write_results_func_with_vast_memory(const std::string& saved_path) {
	while (true) {
        std::vector<InferResult> results;

        {
            std::unique_lock<std::mutex> lock(resultMutex);
            resultCondVar.wait(lock, [] { return infer_done || !resultQueue.empty() || (preprocess_done && inputQueueV2.empty()); });

            if (resultQueue.empty() && infer_done && inputQueueV2.empty() && preprocess_done) {
                std::cout << "Write results func finished!" << std::endl; 
                break;
            }
            // collect all results
            results.reserve(resultQueue.size());
            while  (!resultQueue.empty()) {
                InferResult& infer_result = resultQueue.front();
                results.emplace_back(std::move(infer_result));
                resultQueue.pop();
            }
        }
        
        // std::cout << "result size: " << results.size() << std::endl;
        for (int i = 0; i < results.size(); ++i) {
            InferResult& infer_result = results[i];
            // write results to save path
                std::string file_name = seeta::getFileName(infer_result.image);
                std::string base_name = seeta::getBaseName(file_name);
                std::string saved_txt = saved_path + "/" + base_name + ".txt";
                std::ofstream out(saved_txt);
                for (int j = 0; j < infer_result.results.size(); ++j) {
                    float score = infer_result.results[j].score;
                    bbox box = infer_result.results[j].box;
                    int cls = infer_result.results[j].cls;
                    if (j != 0) out << std::endl;
                    out << j + 1 << " " << cls << " " << score 
                        << " " << box.x << " " << box.y
                        << " " << box.x + box.width << " " << box.y
                        << " " << box.x + box.width << " " << box.y + box.height
                        << " " << box.x << " " << box.y + box.height
                        << " " << box.x + box.width / 2.0 << " " << box.y + box.height / 2.0;
                }
                out.close();
        }

	}
}


static void write_results_func_with_vast_memory_with_thread_pool(const std::string& saved_path, 
                                otl::ThreadPool& thread_pool) {
	while (true) {
        std::vector<InferResult> results;

        {
            std::unique_lock<std::mutex> lock(resultMutex);
            resultCondVar.wait(lock, [] { return infer_done || !resultQueue.empty() || (preprocess_done && inputQueueV2.empty()); });

            if (resultQueue.empty() && infer_done && inputQueueV2.empty() && preprocess_done) {
                std::cout << "Write results func finished!" << std::endl; 
                break;
            }
            // collect all results
            results.reserve(resultQueue.size());
            while  (!resultQueue.empty()) {
                InferResult& infer_result = resultQueue.front();
                results.emplace_back(std::move(infer_result));
                resultQueue.pop();
            }
        }
        
        // std::cout << "result size: " << results.size() << std::endl;
        for (int i = 0; i < results.size(); ++i) {
            InferResult& infer_result = results[i];
            thread_pool.run([&infer_result, &saved_path](int idx){
                // write results to save path
                std::string file_name = seeta::getFileName(infer_result.image);
                std::string base_name = seeta::getBaseName(file_name);
                std::string saved_txt = saved_path + "/" + base_name + ".txt";
                std::ofstream out(saved_txt);
                for (int j = 0; j < infer_result.results.size(); ++j) {
                    float score = infer_result.results[j].score;
                    bbox box = infer_result.results[j].box;
                    int cls = infer_result.results[j].cls;
                    if (j != 0) out << std::endl;
                    out << j + 1 << " " << cls << " " << score 
                        << " " << box.x << " " << box.y
                        << " " << box.x + box.width << " " << box.y
                        << " " << box.x + box.width << " " << box.y + box.height
                        << " " << box.x << " " << box.y + box.height
                        << " " << box.x + box.width / 2.0 << " " << box.y + box.height / 2.0;
                }
                out.close();
            });
            
        }
        
        // wait saving to finish
        thread_pool.join();
	}
}

int main_images_multi_threads_and_producer_consumer(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    Config config =  ReadConfig("config.ini");
    std::cout << config << std::endl;

    std::string images_path = config.parameter.image_path;
    std::string saved_path = config.parameter.save_path;
    if (!seeta::directory_exists(saved_path)) {
        std::cout << "Creating directory " << saved_path << std::endl;
        seeta::create_directory(saved_path);
    }

    std::vector<std::string> images = seeta::FindFilesRecursively(images_path,-1);
    std::cout << "Found " << images.size() << " images." << std::endl;

    std::vector<std::unique_ptr<seeta::Rtdetr, RedetrDeleter>> rtdetrs(config.parameter.workers_num);

    // for (int num = 0; num < config.parameter.workers_num; ++num) {
    //     rtdetrs[num].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
    //                                     config.parameter.detector_thresh));
    // }

    // init using thread pool
    otl::ThreadPool thread_pool(config.parameter.workers_num);
    for (int i = 0; i < config.parameter.workers_num; i++) {
        thread_pool.run([&rtdetrs, &config](int idx){
        rtdetrs[idx].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
                                        config.parameter.detector_thresh));
        });
    }

    thread_pool.join();

    int images_size = images.size();
    int input_size = rtdetrs[0]->input_dims().d[2];


    std::thread preprocess_thread(preprocess_func, std::ref(images_path), std::ref(images), 
                                std::ref(config), input_size);

	std::thread inference_thread(infer_func, std::ref(rtdetrs), std::ref(thread_pool),
                            std::ref(config));
	std::thread write_thread(write_results_func, std::ref(saved_path));

	preprocess_thread.join();
	inference_thread.join();
	write_thread.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Processing " << images_size << " images spent " 
            << duration.count() * 1.0 << "ms" << std::endl;

    return 0;
}

int main_images_multi_threads_and_producer_consumer_with_vast_memory(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    Config config =  ReadConfig("config.ini");
    std::cout << config << std::endl;

    std::string images_path = config.parameter.image_path;
    std::string saved_path = config.parameter.save_path;
    if (!seeta::directory_exists(saved_path)) {
        std::cout << "Creating directory " << saved_path << std::endl;
        seeta::create_directory(saved_path);
    }

    std::vector<std::string> images = seeta::FindFilesRecursively(images_path,-1);
    std::cout << "Found " << images.size() << " images." << std::endl;

    std::vector<std::unique_ptr<seeta::Rtdetr, RedetrDeleter>> rtdetrs(config.parameter.workers_num);

    // for (int num = 0; num < config.parameter.workers_num; ++num) {
    //     rtdetrs[num].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
    //                                     config.parameter.detector_thresh));
    // }

    // init using thread pool
    otl::ThreadPool thread_pool(config.parameter.workers_num);
    for (int i = 0; i < config.parameter.workers_num; i++) {
        thread_pool.run([&rtdetrs, &config](int idx){
        rtdetrs[idx].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
                                        config.parameter.detector_thresh));
        });
    }

    thread_pool.join();

    int images_size = images.size();
    int input_size = rtdetrs[0]->input_dims().d[2];

    // init vast memory
    auto start1 = std::chrono::high_resolution_clock::now();
    otl::vast_memory<float> vast_memory(1 * 3 * input_size * input_size, config.parameter.workers_num * 4);
    std::cout << "Vast memory groups:" << config.parameter.workers_num * 4 << ", group_size: " << 1 * 3 * input_size * input_size << std::endl;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;
    std::cout << "Init vast_memory spent " << duration1.count() << "ms" << std::endl; 

    std::thread preprocess_thread(preprocess_func_with_vast_memory, std::ref(images_path), std::ref(images), 
                                std::ref(config), input_size, std::ref(vast_memory));

	std::thread inference_thread(infer_func_with_vast_memory, std::ref(rtdetrs), std::ref(thread_pool),
                            std::ref(config), std::ref(vast_memory));
	std::thread write_thread(write_results_func_with_vast_memory, std::ref(saved_path));

	preprocess_thread.join();
	inference_thread.join();
	write_thread.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Processing " << images_size << " images spent " 
            << duration.count() * 1.0 << "ms" << std::endl;

    return 0;
}



int main_images_multi_threads_and_producer_consumer_with_vast_memory_with_multi_saver(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    Config config =  ReadConfig("config.ini");
    std::cout << config << std::endl;

    std::string images_path = config.parameter.image_path;
    std::string saved_path = config.parameter.save_path;
    if (!seeta::directory_exists(saved_path)) {
        std::cout << "Creating directory " << saved_path << std::endl;
        seeta::create_directory(saved_path);
    }

    std::vector<std::string> images = seeta::FindFilesRecursively(images_path,-1);
    std::cout << "Found " << images.size() << " images." << std::endl;

    std::vector<std::unique_ptr<seeta::Rtdetr, RedetrDeleter>> rtdetrs(config.parameter.workers_num);

    // for (int num = 0; num < config.parameter.workers_num; ++num) {
    //     rtdetrs[num].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
    //                                     config.parameter.detector_thresh));
    // }

    // init using thread pool
    otl::ThreadPool thread_pool(config.parameter.workers_num);
    otl::ThreadPool saver_thread_pool(config.parameter.saver_num);
    for (int i = 0; i < config.parameter.workers_num; i++) {
        thread_pool.run([&rtdetrs, &config](int idx){
        rtdetrs[idx].reset(new seeta::Rtdetr(config.model.detector_model.c_str(), 
                                        config.parameter.detector_thresh));
        });
    }

    thread_pool.join();

    int images_size = images.size();
    int input_size = rtdetrs[0]->input_dims().d[2];

    // init vast memory
    auto start1 = std::chrono::high_resolution_clock::now();
    otl::vast_memory<float> vast_memory(1 * 3 * input_size * input_size, config.parameter.workers_num * 4);
    std::cout << "Vast memory groups:" << config.parameter.workers_num * 4 << ", group_size: " << 1 * 3 * input_size * input_size << std::endl;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;
    std::cout << "Init vast_memory spent " << duration1.count() << "ms" << std::endl; 

    std::thread preprocess_thread(preprocess_func_with_vast_memory, std::ref(images_path), std::ref(images), 
                                std::ref(config), input_size, std::ref(vast_memory));

	std::thread inference_thread(infer_func_with_vast_memory, std::ref(rtdetrs), std::ref(thread_pool),
                            std::ref(config), std::ref(vast_memory));
	std::thread write_thread(write_results_func_with_vast_memory_with_thread_pool, 
                        std::ref(saved_path), std::ref(saver_thread_pool));

	preprocess_thread.join();
	inference_thread.join();
	write_thread.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Processing " << images_size << " images spent " 
            << duration.count() * 1.0 << "ms" << std::endl;

    return 0;
}

int main(int argc, char** argv) {
    // return main_test(argc, argv);

    if (argc != 2) {
        std::cout << "Usage: test pattern_code." << std::endl;
        std::cout << "pattern_code == 1: Simple loop to [preprocess images, \
                infer images and save results]." << std::endl;
        std::cout << "pattern_code == 2: Using thread pool to [preprocess images, \
                infer images and save results]." << std::endl;
        std::cout << "pattern_code == 3: Using independent thread to [preprocess images], \
                    using thread_pool to [infer images], using independent thread to [save results]." << std::endl;
        std::cout << "pattern_code == 4: pattern_code==3 with preallocated memories \
                    to [preprocess image]." << std::endl;
        std::cout << "pattern_code == 5: pattern_code==4 with thread pool to [save results]." << std::endl;
        return 0;
    }
    int pattern_code = atoi(argv[1]);
    std::cout << "code: " << pattern_code << std::endl;
    if (pattern_code == 1)
    {   
        std::cout << std::endl;
        std::cout << "pattern_code == 1: Simple loop to [preprocess images, \
                infer images and save results]." << std::endl;
        return main_images_test(argc, argv);
    }

    if (pattern_code == 2) {
        std::cout << std::endl;
        std::cout << "pattern_code == 2: Using thread pool to [preprocess images, \
                infer images and save results]." << std::endl;
        return main_images_multi_threads(argc, argv);
    }

    if (pattern_code == 3) {
        std::cout << std::endl;
        std::cout << "pattern_code == 3: Using independent thread to [preprocess images], \
                    using thread_pool to [infer images], using independent thread to [save results]." << std::endl;
        return main_images_multi_threads_and_producer_consumer(argc, argv);
    }

    if (pattern_code == 4) {
        std::cout << std::endl;
        std::cout << "pattern_code == 4: pattern_code==3 with preallocated memories \
            to [preprocess image]." << std::endl;
        return main_images_multi_threads_and_producer_consumer_with_vast_memory(argc, argv);
    }

    if (pattern_code == 5) {
        std::cout << std::endl;
        std::cout << "pattern_code == 5: pattern_code==4 with thread pool to [save results]." << std::endl;
        return main_images_multi_threads_and_producer_consumer_with_vast_memory_with_multi_saver(argc, argv);
    }

    return main_image_test(argc, argv);
}