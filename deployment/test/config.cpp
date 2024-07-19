#include "config.h"

#include "ini/iniparser.h"
#include <iostream>

std::ostream& operator<<(std::ostream& out, const Config& cfg)
{

	out << "Detector model: " << cfg.model.detector_model << std::endl;
	
	out << "Images path: " << cfg.parameter.image_path << std::endl;
	out << "Save results to: " << cfg.parameter.save_path << std::endl;
	out << "Detector thresh: " << cfg.parameter.detector_thresh << std::endl;
	out << "Workers num:" << cfg.parameter.workers_num << std::endl;
	// out << "Model input size: " << cfg.parameter.input_size << std::endl;
	out << "Saver num: " << cfg.parameter.saver_num << std::endl;
	out << std::endl;
	return out;
}

const Config ReadConfig(const std::string& ininame)
{
	dictionary *ini = iniparser_load(ininame.c_str());
	if (!ini)
	{
		std::string msg = "Can not load ini: " + ininame;
		std::cerr << msg << std::endl;
		throw std::logic_error(msg);
	}

	Config cfg;
	cfg.model.detector_model = iniparser_getstring(ini, "model:DETECTOR_MODEL", "null");

	cfg.parameter.image_path = iniparser_getstring(ini, "parameter:IMAGE_PATH","null");
	cfg.parameter.save_path = iniparser_getstring(ini, "parameter:SAVE_PATH", "null");

	cfg.parameter.detector_thresh = iniparser_getdouble(ini, "parameter:DETECTOR_THRESH", 0.0);
	cfg.parameter.workers_num = iniparser_getint(ini, "parameter:WORKERS_NUM", 1);
	// cfg.parameter.input_size = iniparser_getint(ini, "parameter:INPUT_SIZE", 640);
	cfg.parameter.saver_num = iniparser_getint(ini, "parameter:SAVER_NUM", 1);
	iniparser_freedict(ini);

	return cfg;
}