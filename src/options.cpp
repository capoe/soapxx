#include "options.hpp"


namespace soap {
    

void Options::registerPython() {
	using namespace boost::python;
	void (Options::*set_int)(std::string, int) = &Options::set;
	void (Options::*set_double)(std::string, double) = &Options::set;
	void (Options::*set_string)(std::string, std::string) = &Options::set;

	class_<Options>("Options")
	    .def("configureRealBasis", &Options::configureRealBasis)
		.def("configureReciprocalBasis", &Options::configureReciprocalBasis)
	    .def("summarizeOptions", &Options::summarizeOptions)
		.def("configureCenters", &Options::configureCenters)
		.def("configureReciprocalLattice", &Options::configureReciprocalLattice)
		.def("set", set_int)
		.def("set", set_double)
		.def("set", set_string);
}

} /* Close namespaces */
