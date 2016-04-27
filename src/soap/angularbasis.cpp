#include "soap/angularbasis.hpp"

#include <math.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>

namespace soap {


void AngularBasis::configure(Options &options) {
    _L = options.get<int>("angularbasis.L");
}

void AngularBasis::computeCoefficients(
        vec d,
        double r,
        double sigma,
        angcoeff_t &Ylm,
        angcoeff_t *dYlm_dx,
        angcoeff_t *dYlm_dy,
        angcoeff_t *dYlm_dz) {


    if (r < AngularBasis::RADZERO) {
        // SCALARS
        std::complex<double> c00 = boost::math::spherical_harmonic<double,double>(0, 0, 0.0, 0.0);
        Ylm(0) = c00;
        // GRADIENTS
        if (dYlm_dx) {
            assert(dYlm_dy != NULL && dYlm_dz != NULL);
            ; // <- all zero
        }
    }
    else {
		double theta = acos(d.getZ());
		double phi = atan2(d.getY(), d.getX());
		if (phi < 0.) phi += 2*M_PI; // <- Shift [-pi, -0] to [pi, 2*pi]
		// SCALARS
		for (int l = 0; l <= _L; ++l) {
			for (int m = -l; m <= l; ++m) {
				std::complex<double> clm = boost::math::spherical_harmonic<double,double>(l, m, theta, phi);
				Ylm(l*l+l+m) = clm;
			}
		}
		// GRADIENTS
        if (dYlm_dx) {
            assert(dYlm_dy != NULL && dYlm_dz != NULL);
            vec dr = r*d;
            for (int l = 0; l <= _L; ++l) {
                for (int m = -l; m <= l; ++m) {
                    int lm = l*l+l+m;
                    std::vector<std::complex<double> > dYlm = GradSphericalYlm::eval(l, m, dr);
                    (*dYlm_dx)(lm) = dYlm[0];
                    (*dYlm_dy)(lm) = dYlm[1];
                    (*dYlm_dz)(lm) = dYlm[2];
                }
            }
        }
    }

    return;
}

void AngularBasisFactory::registerAll(void) {
	AngularBasisOutlet().Register<AngularBasis>("spherical-harmonic");
}

} /* CLOSE NAMESPACE */

