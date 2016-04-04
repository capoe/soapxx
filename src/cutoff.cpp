#include "cutoff.hpp"

namespace soap {


void CutoffFunction::configure(Options &options) {
    _Rc = options.get<double>("radialcutoff.Rc");
    _Rc_width = options.get<double>("radialcutoff.Rc_width");
    _center_weight = options.get<double>("radialcutoff.center_weight");

    GLOG() << "Weighting function with "
    	<< "Rc = " << _Rc
    	<< ", _Rc_width = " << _Rc_width
		<< ", central weight = " << _center_weight << std::endl;
}

void CutoffFunctionFactory::registerAll(void) {
	CutoffFunctionOutlet().Register<CutoffFunction>("shifted-cosine");
}


}

