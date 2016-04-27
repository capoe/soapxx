#include "soap/cutoff.hpp"

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

bool CutoffFunction::isWithinCutoff(double r) {
    return (r <= _Rc);
}

double CutoffFunction::calculateWeight(double r) {
    double weight_at_r = 1.;
    if (r > _Rc) {
        weight_at_r = -1.e-10;
    }
    else if (r <= _Rc - _Rc_width) {
        weight_at_r = 1.;
    }
    else {
        weight_at_r = 0.5*(1+cos(M_PI*(r-_Rc+_Rc_width)/_Rc_width));
        //weight_at_r = pow(cos(0.5*M_PI*(r-_Rc+_Rc_width)/_Rc_width),2);
    }
    return weight_at_r;
}

vec CutoffFunction::calculateGradientWeight(double r, vec d) {
    vec grad_weight(0.,0.,0.);
    if (r > _Rc) {
        ;
    }
    else if (r <= _Rc - _Rc_width) {
        ;
    }
    else {
        grad_weight = -0.5*sin(M_PI*(r-_Rc+_Rc_width)/_Rc_width)*M_PI/_Rc_width * d;
        //double phi = 0.5*M_PI*(r-_Rc+_Rc_width)/_Rc_width;
        //grad_weight = - M_PI/_Rc_width*sin(phi)*cos(phi) * d;
    }
    return grad_weight;
}

void CutoffFunctionFactory::registerAll(void) {
	CutoffFunctionOutlet().Register<CutoffFunction>("shifted-cosine");
}


}

