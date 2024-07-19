#pragma once
#include <string>
#include <vector>

struct Config
{
	struct
	{
		std::string detector_model;

	} model;

	struct
	{
		std::string image_path;
		std::string save_path;

		float detector_thresh;
		int workers_num;
		// int input_size;
		int saver_num;
	} parameter;

};

std::ostream &operator<<(std::ostream &out, const Config &cfg);

const Config ReadConfig(const std::string &ininame);
