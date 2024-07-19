#include "rtdetr.h"
#include "rtdetr_utils.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

namespace seeta {

    class Logger : public ILogger {
        void log(Severity severity, const char* msg) noexcept override {

            if (severity < Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    };

    static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz) 
    {
        unsigned char* data;
        int            ret;

        data = NULL;

        if (NULL == fp) {
            return NULL;
        }

        ret = fseek(fp, ofst, SEEK_SET);
        if (ret != 0) {
            printf("blob seek failure.\n");
            return NULL;
        }

        data = (unsigned char*)malloc(sz);
        if (data == NULL) {
            printf("buffer malloc failure.\n");
            return NULL;
        }
        ret = fread(data, 1, sz, fp);
        return data;
    }

    static unsigned char* load_model(const char* filename, int* model_size)
    {
        FILE*          fp;
        unsigned char* data;

        fp = fopen(filename, "rb");
        if (NULL == fp) {
            printf("Open file %s failed.\n", filename);
            return NULL;
        }

        fseek(fp, 0, SEEK_END);
        int size = ftell(fp);

        data = load_data(fp, 0, size);

        fclose(fp);

        *model_size = size;
        return data;
    }


    Rtdetr::Rtdetr(const char* engine_file, float confidence_thresh) {
        m_conf_thresh = confidence_thresh;

        // init logger
        m_logger.reset(new Logger());
        // std::cout << "logger init succeed." << std::endl;

        // init runtime
        m_runtime = createInferRuntime(*m_logger.get());
        if(!m_runtime) std::cout << "runtime init failed." << std::endl;
        // std::cout << "runtime init succeed." << std::endl;

        int model_size;
        unsigned char* model_data = load_model(engine_file, &model_size);

        // init engine
        m_engine = m_runtime->deserializeCudaEngine(model_data, model_size);
        if (!m_engine) std::cout << "engine init failed." << std::endl;
        // std::cout<< "engine init succeed." << std::endl;

        // init context
        m_context = m_engine->createExecutionContext();
        if (!m_context) std::cout << "context init failed." << std::endl;
        // std::cout << "context init succeed." << std::endl;

        // get input output shape
        int io_number = m_engine->getNbBindings();
        assert(io_number == 2);
        if (m_engine->bindingIsInput(0)) {
            m_input_dims = m_context->getBindingDimensions(0);
        }
        if (!m_engine->bindingIsInput(1)) {
            m_output_dims = m_context->getBindingDimensions(1);
        }
        // std::cout << "input dims:";
        for (int i = 0; i < m_input_dims.nbDims; ++i) {
            m_cuda_input_size *= m_input_dims.d[i];
            // std::cout << m_input_dims.d[i];
            // if (i < m_input_dims.nbDims - 1) {
            //     std::cout << "x";
            // }
        }
        // std::cout << std::endl;
        // std::cout << "cuda input size: " << m_cuda_input_size << std::endl;

        // std::cout << "output dims:";
        for (int i = 0; i < m_output_dims.nbDims; ++i) {
            m_cuda_output_size *= m_output_dims.d[i];
            // std::cout << m_output_dims.d[i];
            // if (i < m_output_dims.nbDims - 1) {
            //     std::cout << "x";
            // }
        }
        // std::cout << std::endl;
        // std::cout << "cuda output size: " << m_cuda_output_size << std::endl;

        // alloc mem for cuda and host
        cudaMalloc(&m_cuda_input_mem, 1 * m_cuda_input_size * sizeof(float));
        cudaMalloc(&m_cuda_output_mem, 1 * m_cuda_output_size * sizeof(float));
        // std::cout << "m_cuda_input_mem ptr:" << m_cuda_input_mem << std::endl;
        // std::cout << "m_cuda_ouput_mem ptr:" << m_cuda_output_mem << std::endl;

        
        // auto deleter = [](float* data) {if (data) delete[] data;};
        // m_host_input_mem.reset(new float[1 * m_cuda_input_size]);
        // m_host_output_mem.reset(new float[1 * m_cuda_output_size]);
        cudaMallocHost((void**)&m_host_input_mem, 1 * m_cuda_input_size * sizeof(float));
        cudaMallocHost((void**)&m_host_output_mem, 1 * m_cuda_output_size * sizeof(float));

        // std::cout << "m_host_input_mem ptr:" << m_host_input_mem << std::endl;
        // std::cout << "m_host_output_mem ptr:" << m_host_output_mem << std::endl;

        // free model_data
        if(model_data) {
            free(model_data);
        }
    }

    Rtdetr::~Rtdetr() {
        // free cuda malloc memory
        if (m_cuda_input_mem)
            cudaFree(m_cuda_input_mem);

        if (m_cuda_output_mem)
            cudaFree(m_cuda_output_mem);

        // free host memory
        if (m_host_input_mem)
            cudaFreeHost(m_host_input_mem);

        if (m_host_output_mem)
            cudaFreeHost(m_host_output_mem);

        // runtime engine contest
        m_context->destroy();
        m_engine->destroy();
        m_runtime->destroy();
    }

    static std::vector<float> cxcywh_to_xyxy(const std::vector<float>& box) {
        float x1 = box[0] - box[2] / 2.0f;
        float y1 = box[1] - box[3] / 2.0;
        float x2 = box[0] + box[2] / 2.0f;
        float y2 = box[1] + box[3] / 2.0f;
        std::vector<float> results = {x1, y1, x2, y2};
        return results;
    }

    // post processing to get results
    // raw_output num_queries x (4 + cls_num)
    static void postprocess(float* raw_output, int num_queries, int cls_num, int origin_image_width, 
                int origin_image_height, float conf_thresh, std::vector<detect_result>& results) 
    {
        // results.clear();
        // std::cout << "num_queries: " << num_queries << std::endl;
        // std::cout << "cls_num: " << cls_num << std::endl;
        // for (int i=0;i<3;++i) {
        //     std::cout << i << "th output:" << raw_output[0 * 84 + i] << " ";
        // }

        for (int i = 0; i < num_queries; ++i) {
            float* output = raw_output + i * (4 + cls_num);
            float cx = output[0];
            float cy = output[1];
            float width = output[2];
            float height = output[3];
            float* scores = output + 4;
            int max_idx = 0;
            float max_score = 0.0f;

            for (int j = 0; j < cls_num; j++) {
                // std::cout << "score " << j << ":" << scores[j] << " ";
                if (scores[j] > max_score) {
                    max_score = scores[j];
                    max_idx = j;
                }
            }
            // std::cout <<std::endl;

            // only collect result which confidence is greater than thresh
            // std::cout << "max_score: " << max_score << std::endl;
            if (max_score >= conf_thresh) {
                detect_result result;
                result.score = max_score;
                result.cls = max_idx;
                std::vector<float> xyxy = cxcywh_to_xyxy(std::vector<float>{cx, cy, width, height});
                // decode location
                xyxy[0] = std::min(std::max(0.0f, xyxy[0] * origin_image_width), origin_image_width - 1.0f);
                xyxy[1] = std::min(std::max(0.0f, xyxy[1] * origin_image_height), origin_image_height - 1.0f);
                xyxy[2] = std::min(std::max(0.0f, xyxy[2] * origin_image_width), origin_image_width - 1.0f);
                xyxy[3] = std::min(std::max(0.0f, xyxy[3] * origin_image_height), origin_image_height - 1.0f);
                result.box.x = (xyxy[0]);
                result.box.y = (xyxy[1]);
                result.box.width = (xyxy[2] - xyxy[0]);
                result.box.height = (xyxy[3] - xyxy[1]);
                // std::cout << "obj width:" << result.box.width << std::endl;
                // std::cout << "obj height:" << result.box.height << std::endl;
                if ((result.box.width > 0) && (result.box.height > 0)) {
                    results.emplace_back(result);
                }
            }

        }
    }

    detect_result_group Rtdetr::detect(unsigned char* image, int image_width, int image_height, bool debug) {
        cv::Mat origin_mat(image_height, image_width, CV_8UC3, (void*)image);
        float scale_x,scale_y;
        int padding_top, padding_bottom, padding_left, padding_right;
        {
            auto start = std::chrono::high_resolution_clock::now();
            seeta::preprocess(origin_mat, m_input_dims.d[3], m_input_dims.d[2],
                        scale_x, scale_y, padding_top, padding_bottom, padding_left, padding_right, 
                        true, (float*)m_host_input_mem);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            if (debug)
                std::cout << "processing spent " << duration.count() << "ms" << std::endl; 
        }

        // copy host data to cuda
        cudaMemcpy(m_cuda_input_mem, (void*)m_host_input_mem, 1 * m_cuda_input_size * sizeof(float), cudaMemcpyHostToDevice);
        
        void* bindings[] = {m_cuda_input_mem, m_cuda_output_mem};
        // inference
        {
            auto start = std::chrono::high_resolution_clock::now();
            // async running
            // m_context->enqueueV2(bindings, 0, nullptr);

            // sync running 
            m_context->executeV2(bindings);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            if (debug)
                std::cout << "inference spent " << duration.count() << "ms" << std::endl; 
        }

        // copy cuda to host
        // std::cout << "Begin to copy results to host." << std::endl;
        cudaMemcpy((void*)m_host_output_mem, m_cuda_output_mem, 1 * m_cuda_output_size * sizeof(float), cudaMemcpyDeviceToHost);
        // std::cout << "Copy output succeed." << std::endl;

        // clear results before decode
        m_results.clear();
        {
            auto start = std::chrono::high_resolution_clock::now();
            postprocess((float*)m_host_output_mem, m_output_dims.d[1], m_output_dims.d[2] - 4, 
                image_width, image_height, m_conf_thresh, m_results);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            if (debug)
                std::cout << "postprocessing spent " << duration.count() << "ms" << std::endl; 
        }

        detect_result_group result_group;
        result_group.size = m_results.size();
        result_group.data = m_results.data();
        return result_group;
    }

    std::vector<detect_result> Rtdetr::detect(float* chw_data, int image_width, int image_height) {
        // copy host data to cuda
        cudaMemcpy(m_cuda_input_mem, (void*)chw_data, 1 * m_cuda_input_size * sizeof(float), cudaMemcpyHostToDevice);
        void* bindings[] = {m_cuda_input_mem, m_cuda_output_mem};
        // inference
        // sync running 
        m_context->executeV2(bindings);
        // copy cuda to host
        cudaMemcpy((void*)m_host_output_mem, m_cuda_output_mem, 1 * m_cuda_output_size * sizeof(float), cudaMemcpyDeviceToHost);
        // clear results before decode
        m_results.clear();
        {
            postprocess((float*)m_host_output_mem, m_output_dims.d[1], m_output_dims.d[2] - 4, 
                image_width, image_height, m_conf_thresh, m_results);
        }
        return m_results;
    }

    nvinfer1::Dims Rtdetr::input_dims() const {
        return m_input_dims;
    }
}