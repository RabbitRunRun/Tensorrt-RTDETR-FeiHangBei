#ifndef RTDETR_H_
#define RTDETR_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <vector>
#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "NvInfer.h"
using namespace nvinfer1;

#define API_EXPORT __attribute__((visibility("default")))


// (x,y) means left top position
struct bbox {
    float x;
    float y;
    float width;
    float height;
};

struct detect_result {
    bbox box;
    float score;
    int cls;
};

struct detect_result_group {
    int size;
    detect_result* data;
};

namespace seeta {
    
    struct InferDeleter
    {
        template <typename T>
        void operator()(T* obj) const
        {
            delete obj;
        }
    };

    struct ArrayDeleter {
        void operator()(float* ptr) const {
            if (ptr)
            delete[] ptr;
        }
    };

    class Rtdetr {
        public:
            API_EXPORT Rtdetr(const char* engine_file, float confidence_thresh);
            API_EXPORT ~Rtdetr();

            API_EXPORT detect_result_group detect(unsigned char* image, int image_width, int image_height, bool debug=false);
            API_EXPORT std::vector<detect_result> detect(float* chw_data, int image_width, int image_height);
            API_EXPORT nvinfer1::Dims input_dims() const;
            API_EXPORT Rtdetr(const Rtdetr&) = delete;
            API_EXPORT Rtdetr(Rtdetr&&) = delete;
            API_EXPORT Rtdetr& operator=(const Rtdetr&) = delete;
            API_EXPORT Rtdetr& operator=(Rtdetr&&) = delete;
        private:
            nvinfer1::IRuntime* m_runtime;
            nvinfer1::ICudaEngine* m_engine;
            nvinfer1::IExecutionContext* m_context;
            std::unique_ptr<nvinfer1::ILogger, InferDeleter> m_logger;

            nvinfer1::Dims m_input_dims;
            nvinfer1::Dims m_output_dims;
            void* m_cuda_input_mem;
            int m_cuda_input_size = 1;

            void* m_cuda_output_mem;
            int m_cuda_output_size = 1;
            

            // std::unique_ptr<float[], ArrayDeleter> m_host_input_mem;
            // std::unique_ptr<float[], ArrayDeleter> m_host_output_mem;
            void* m_host_input_mem;
            void* m_host_output_mem;

            float m_conf_thresh;
            std::vector<detect_result> m_results;
            
    };
}


#ifdef _cplusplus
}
#endif

#endif // RTDETR_H_
