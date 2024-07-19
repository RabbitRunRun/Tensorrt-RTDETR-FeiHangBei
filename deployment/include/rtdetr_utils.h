#include <dirent.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>
#include <memory>
#include <queue>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace seeta {
	static const std::string FileSeparator() {
#if ORZ_PLATFORM_OS_WINDOWS
		return "\\";
#else
		return "/";
#endif
	}

static std::string getFileName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

static std::string getBaseName(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) {
        return filename;
    }
    return filename.substr(0, pos);
}

	static std::vector<std::string> FindFilesCore(const std::string &path, std::vector<std::string> *dirs = nullptr) {
		std::vector<std::string> result;
		if (dirs) dirs->clear();
#if ORZ_PLATFORM_OS_WINDOWS
		_finddata_t file;
		std::string pattern = path + FileSeparator() + "*";
		auto handle = _findfirst(pattern.c_str(), &file);

		if (handle == -1L) return result;
		do {
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0) continue;
			if (file.attrib & _A_SUBDIR) {
				if (dirs) dirs->push_back(file.name);
			}
			else {
				result.push_back(file.name);
			}
		} while (_findnext(handle, &file) == 0);

		_findclose(handle);
#else
		struct dirent *file;

		auto handle = opendir(path.c_str());

		if (handle == nullptr) return result;

		while ((file = readdir(handle)) != nullptr)
		{
			if (strcmp(file->d_name, ".") == 0 || strcmp(file->d_name, "..") == 0) continue;
			if (file->d_type & DT_DIR)
			{
				if (dirs) dirs->push_back(file->d_name);
			}
			else if (file->d_type & DT_REG)
			{
				result.push_back(file->d_name);
			}
			// DT_LNK // for linkfiles
		}

		closedir(handle);
#endif
		return std::move(result);
	}

	static std::vector<std::string> FindFiles(const std::string &path) {
		return FindFilesCore(path);
	}

	static std::vector<std::string> FindFiles(const std::string &path, std::vector<std::string> &dirs) {
		return FindFilesCore(path, &dirs);
	}

    // 递归获取文件
	static std::vector<std::string> FindFilesRecursively(const std::string &path, int depth) {
		std::vector<std::string> result;
		std::queue<std::pair<std::string, int> > work;
		std::vector<std::string> dirs;
		std::vector<std::string> files = FindFiles(path, dirs);
		result.insert(result.end(), files.begin(), files.end());
		for (auto &dir : dirs) work.push({ dir, 1 });
		while (!work.empty()) {
			auto local_pair = work.front();
			work.pop();
			auto local_path = local_pair.first;
			auto local_depth = local_pair.second;
			if (depth > 0 && local_depth >= depth) continue;
			files = FindFiles(path + FileSeparator() + local_path, dirs);
			for (auto &file : files) result.push_back(local_path + FileSeparator() + file);
			for (auto &dir : dirs) work.push({ local_path + FileSeparator() + dir, local_depth + 1 });
		}
		return result;
	}

    // 检查文件夹是否存在
    static bool directory_exists(const std::string& path) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            return false;
        } else if (info.st_mode & S_IFDIR) {
            return true;
        } else {
            return false;
        }
    }

    // 创建文件夹
    static bool create_directory(const std::string& path) {
        if (mkdir(path.c_str(), 0777) == 0) {
            return true;
        } else {
            std::cerr << "create path " << path << "failed." << std::endl;
            return false;
        }
    }

    // letter box
    static cv::Mat letter_box(const cv::Mat &origin_mat,int model_input_width, 
                        int model_input_height, float& scale_x, float& scale_y, int& padding_top, int& padding_bottom, 
                        int& padding_left, int& padding_right, bool scale_fill)
    {   
        int image_width = origin_mat.cols;
        int image_height = origin_mat.rows;
        if (scale_fill) {
            // just stretch

            scale_x = model_input_width * 1.0 / image_width;
            scale_y = model_input_height * 1.0 / image_height;
            padding_top = 0;
            padding_bottom = 0;
            padding_left = 0;
            padding_right = 0;

        } else {
            float scale = std::min(model_input_width * 1.0 / image_width, model_input_height * 1.0 / image_height);

            scale_x = scale;
            scale_y = scale;
            // padding a image
            // center padding
            float dw = (model_input_width - scale * image_width) / 2.0;
            float dh = (model_input_height - scale * image_height) / 2.0;
            padding_top = int(std::round(dh - 0.1));
            padding_bottom = int(std::round(dh + 0.1));
            padding_left = int(std::round(dw - 0.1));
            padding_right = int(std::round(dw + 0.1));
        }
        

         // resize image
        cv::Mat resized_mat;
        cv::resize(origin_mat, resized_mat, cv::Size(int(image_width * scale_x), int(image_height * scale_y)), cv::INTER_LINEAR);

        cv::Mat padded_image;
        cv::copyMakeBorder(resized_mat, padded_image, padding_top, padding_bottom, padding_left, padding_right, 
                cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        return padded_image;
    }

    static bool preprocess(const cv::Mat& origin_mat, int model_input_width, int model_input_height,
                                float& scale_x, float&scale_y, int&padding_top, int&padding_bottom,
                                int& padding_left, int& padding_right, bool scale_fill, float* chw_data) 
    {
        // resize and pad
        cv::Mat resized_border_mat = letter_box(origin_mat, model_input_width, model_input_height,
                                        scale_x, scale_y, padding_top, padding_bottom, padding_left, padding_right, scale_fill);

        // bgr to rgb
        cv::Mat rgb_mat;
        cv::cvtColor(resized_border_mat, rgb_mat, cv::COLOR_BGR2RGB);

        // normalize
        cv::Mat normalized_mat;
        rgb_mat.convertTo(normalized_mat, CV_32FC3, 1.0 / 255);
        
        // hwc to chw
        int channels = normalized_mat.channels();
        int height = normalized_mat.rows;
        int width = normalized_mat.cols;

        // hwc to chw
        // uint8_t* hwc_data = normalized_mat.data;
        // for (int c = 0; c < channels; c++) {
        //     for (int h = 0; h < height; h++) {
        //         for (int w = 0; w < width; w++) {
        //             int dst = c * height * width + h * width + w;
        //             int src = h * width * channels + w * channels + c;
        //             chw_data[dst] = float(hwc_data[src]);
        //         }
        //     }
        // }

        // hwc to chw
        cv::Size float_image_size {width, height};
        std::vector<cv::Mat> chw(channels);
        for (int i = 0; i < channels; ++i) {
            chw[i] = cv::Mat(float_image_size, CV_32FC1, chw_data + i * float_image_size.width * float_image_size.height);
        }
        cv::split(normalized_mat, chw);
        // std::cout << chw_data[0] << " " << chw_data[1] << " " << chw_data[2] << std::endl;

        return true;
    }
}