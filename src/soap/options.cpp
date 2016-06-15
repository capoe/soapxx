#include "soap/options.hpp"


namespace soap {

Options::Options() :
	_center_excludes(boost::python::list()) {

	// Set defaults
    this->set("spectrum.gradients", false);
    this->set("spectrum.2l1_norm", true);
	this->set("radialbasis.type", "gaussian");
	this->set("radialbasis.mode", "equispaced");
	this->set("radialbasis.N", 9);
	this->set("radialbasis.sigma", 0.5);
	this->set("radialbasis.integration_steps", 15);
	this->set("radialcutoff.type", "shifted-cosine");
	this->set("radialcutoff.Rc", 4.);
	this->set("radialcutoff.Rc_width", 0.5);
	this->set("radialcutoff.center_weight", 1.);
	this->set("angularbasis.type", "spherical-harmonic");
	this->set("angularbasis.L", 6);
	this->set("densitygrid.N", 20);
	this->set("densitygrid.dx", 0.15);
}

//template<typename return_t>
//return_t Options::get(std::string key) {
	//return soap::lexical_cast<return_t, std::string>(_key_value_map[key], "wrong or missing type in " + key);
//}

std::string Options::summarizeOptions() {
	std::string info = "";
	info += "Options:\n";
	std::map<std::string, std::string>::iterator it;
	for (it = _key_value_map.begin(); it != _key_value_map.end(); ) {
		info += (boost::format(" o %1$-30s : %2$s") % it->first % it->second).str();
		if (++it != _key_value_map.end()) info += "\n";
	}
	return info;
}

void Options::excludeCenters(boost::python::list &types) {
    for (int i = 0; i < boost::python::len(types); ++i) {
        std::string type = boost::python::extract<std::string>(types[i]);
        _exclude_center[type] = true;
        _exclude_center_list.append(type);
    }
    return;
}

void Options::excludeTargets(boost::python::list &types) {
    for (int i = 0; i < boost::python::len(types); ++i) {
        std::string type = boost::python::extract<std::string>(types[i]);
        _exclude_target[type] = true;
        _exclude_target_list.append(type);
    }
    return;
}

bool Options::doExcludeCenter(std::string &type) {
    auto it = _exclude_center.find(type);
    return (it == _exclude_center.end()) ? false : true;
}

bool Options::doExcludeTarget(std::string &type) {
    auto it = _exclude_target.find(type);
    return (it == _exclude_target.end()) ? false : true;
}

void Options::excludeCenterIds(boost::python::list &types) {
    for (int i = 0; i < boost::python::len(types); ++i) {
        int pid = boost::python::extract<int>(types[i]);
        _exclude_center_id[pid] = true;
        _exclude_center_id_list.append(pid);
    }
    return;
}

void Options::excludeTargetIds(boost::python::list &types) {
    for (int i = 0; i < boost::python::len(types); ++i) {
        int pid = boost::python::extract<int>(types[i]);
        _exclude_target_id[pid] = true;
        _exclude_target_id_list.append(pid);
    }
    return;
}

bool Options::doExcludeCenterId(int pid) {
    auto it = _exclude_center_id.find(pid);
    return (it == _exclude_center_id.end()) ? false : true;
}

bool Options::doExcludeTargetId(int pid) {
    auto it = _exclude_target_id.find(pid);
    return (it == _exclude_target_id.end()) ? false : true;
}

void Options::generateExclusionLists() {
    if (!boost::python::len(_exclude_center_list)) {
        for (auto it = _exclude_center.begin(); it != _exclude_center.end(); ++it) {
            if (it->second) _exclude_center_list.append(it->first);
        }
    }
    if (!boost::python::len(_exclude_target_list)) {
        for (auto it = _exclude_target.begin(); it != _exclude_target.end(); ++it) {
            if (it->second) _exclude_target_list.append(it->first);
        }
    }
    if (!boost::python::len(_exclude_center_id_list)) {
        for (auto it = _exclude_center_id.begin(); it != _exclude_center_id.end(); ++it) {
            if (it->second) _exclude_center_id_list.append(it->first);
        }
    }
    if (!boost::python::len(_exclude_target_id_list)) {
        for (auto it = _exclude_target_id.begin(); it != _exclude_target_id.end(); ++it) {
            if (it->second) _exclude_target_id_list.append(it->first);
        }
    }
    return;
}

void Options::registerPython() {
	using namespace boost::python;
	void (Options::*set_int)(std::string, int) = &Options::set;
	void (Options::*set_double)(std::string, double) = &Options::set;
	void (Options::*set_string)(std::string, std::string) = &Options::set;
	std::string (Options::*get_string)(std::string) = &Options::get;
	//void (Options::*set_bool)(std::string, bool) = &Options::set;

	class_<Options, Options*>("Options")
	    .add_property("exclude_center_list", &Options::getExcludeCenterList)
	    .add_property("exclude_target_list", &Options::getExcludeTargetList)
	    .add_property("exclude_center_id_list", &Options::getExcludeCenterIdList)
	    .add_property("exclude_target_id_list", &Options::getExcludeTargetIdList)
        .def("__str__", &Options::summarizeOptions)
	    .def("summarizeOptions", &Options::summarizeOptions)
		.def("excludeCenters", &Options::excludeCenters)
		.def("excludeTargets", &Options::excludeTargets)
		.def("excludeCenterIds", &Options::excludeCenterIds)
		.def("excludeTargetIds", &Options::excludeTargetIds)
		.def("set", set_int)
		.def("set", set_double)
		.def("set", set_string)
		.def("get", get_string);
}

} /* Close namespaces */
