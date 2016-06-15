#include "soap/cutoff.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

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
    if (r > _Rc || r <= _Rc - _Rc_width) {
        ;
    }
    else {
        grad_weight = -0.5*sin(M_PI*(r-_Rc+_Rc_width)/_Rc_width)*M_PI/_Rc_width * d;
        //double phi = 0.5*M_PI*(r-_Rc+_Rc_width)/_Rc_width;
        //grad_weight = - M_PI/_Rc_width*sin(phi)*cos(phi) * d;
    }
    return grad_weight;
}

void CutoffFunctionHeaviside::configure(Options &options) {
    _Rc = options.get<double>("radialcutoff.Rc");
    _Rc_width = options.get<double>("radialcutoff.Rc_width");
    _center_weight = options.get<double>("radialcutoff.center_weight");

    // To harmonize with adaptive Gaussian basis sets, increase cutoff:
    if (options.hasKey("radialcutoff.Rc_heaviside")) {
        _Rc = options.get<double>("radialcutoff.Rc_heaviside"); // <- Set by RadialBasisGaussian::configure
    }

    GLOG() << "Weighting function with "
        << "Rc = " << _Rc << ", central weight = " << _center_weight << std::endl;
}

bool CutoffFunctionHeaviside::isWithinCutoff(double r) {
    return (r <= _Rc);
}

double CutoffFunctionHeaviside::calculateWeight(double r) {
    return (r > _Rc) ? -1.e-10 : 1.;
}

vec CutoffFunctionHeaviside::calculateGradientWeight(double r, vec d) {
    return vec(0.,0.,0.);
}

void CutoffFunctionFactory::registerAll(void) {
	CutoffFunctionOutlet().Register<CutoffFunction>("shifted-cosine");
	CutoffFunctionOutlet().Register<CutoffFunctionHeaviside>("heaviside");
}

}

BOOST_CLASS_EXPORT_IMPLEMENT(soap::CutoffFunctionHeaviside);

