#include "soap/angularbasis.hpp"

#include <math.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>

namespace soap {


void AngularBasis::configure(Options &options) {
    _L = options.get<int>("angularbasis.L");
}

void AngularBasis::computeCoefficients(vec d, double r, angcoeff_t &save_here) {
    if (r < AngularBasis::RADZERO) {
        std::complex<double> c00 = boost::math::spherical_harmonic<double,double>(0, 0, 0.0, 0.0);
		 save_here(0) = c00;
    }
    else {
    	// Radius not zero => compute moments
		double theta = acos(d.getZ());
		double phi = atan2(d.getY(), d.getX());
		// Shift [-pi, -0] to [pi, 2*pi]
		if (phi < 0.) phi += 2*M_PI;
		for (int l = 0; l <= _L; ++l) {
			for (int m = -l; m <= l; ++m) {
				std::complex<double> clm = boost::math::spherical_harmonic<double,double>(l, m, theta, phi);
				save_here(l*l+l+m) = clm;
			}
		}
    }
    return;
}

void AngularBasisFactory::registerAll(void) {
	AngularBasisOutlet().Register<AngularBasis>("spherical-harmonic");
}

} /* CLOSE NAMESPACE */

