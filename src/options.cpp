#include "options.hpp"


namespace soap {

Options::Options() :
	_center_excludes(boost::python::list()) {

	// Set defaults
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
    }
    return;
}

void Options::excludeTargets(boost::python::list &types) {
    for (int i = 0; i < boost::python::len(types); ++i) {
        std::string type = boost::python::extract<std::string>(types[i]);
        _exclude_target[type] = true;
    }
    return;
}

bool Options::doExcludeCenter(std::string &type) {
    map_exclude_t::iterator it = _exclude_center.find(type);
    return (it == _exclude_center.end()) ? false : true;
}

bool Options::doExcludeTarget(std::string &type) {
    map_exclude_t::iterator it = _exclude_target.find(type);
    return (it == _exclude_target.end()) ? false : true;
}

void Options::registerPython() {
	using namespace boost::python;
	void (Options::*set_int)(std::string, int) = &Options::set;
	void (Options::*set_double)(std::string, double) = &Options::set;
	void (Options::*set_string)(std::string, std::string) = &Options::set;

	class_<Options, Options*>("Options")
        .def("__str__", &Options::summarizeOptions)
	    .def("summarizeOptions", &Options::summarizeOptions)
		.def("excludeCenters", &Options::excludeCenters)
		.def("excludeTargets", &Options::excludeTargets)
		.def("set", set_int)
		.def("set", set_double)
		.def("set", set_string);
}

} /* Close namespaces */
