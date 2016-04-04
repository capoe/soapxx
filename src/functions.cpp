#include "functions.hpp"

#include <boost/math/special_functions/erf.hpp>

namespace soap {

// ======================
// RadialGaussian
// ======================

RadialGaussian::RadialGaussian(double r0, double sigma)
: _r0(r0),
  _sigma(sigma),
  _alpha(1./(2.*sigma*sigma)) {

	// COMPUTE NORMALIZATION S g^2 r^2 dr
	// This normalization is to be used for radial basis functions
    double w = 2*_alpha;
    double W0 = 2*_alpha*_r0;
    _integral_r2_g2_dr =
        1./(4.*pow(w, 2.5))*exp(-w*_r0*_r0)*(
            2*sqrt(w)*W0 +
			sqrt(M_PI)*exp(W0*W0/w)*(w+2*W0*W0)*(
			    1 - boost::math::erf<double>(-W0/sqrt(w))
			)
        );
    _norm_r2_g2_dr = 1./sqrt(_integral_r2_g2_dr);

    // COMPUTE NORMALIZATION S g r^2 dr
    // This normalization is to be used for "standard" radial Gaussians
    w = _alpha;
	W0 = _alpha*_r0;
	_integral_r2_g_dr =
		1./(4.*pow(w, 2.5))*exp(-w*_r0*_r0)*(
			2*sqrt(w)*W0 +
			sqrt(M_PI)*exp(W0*W0/w)*(w+2*W0*W0)*(
				1 - boost::math::erf<double>(-W0/sqrt(w))
			)
		);
	_norm_r2_g_dr = 1./_integral_r2_g_dr;
}

RadialGaussian::RadialGaussian() :
	_r0(0.),
	_sigma(0.),
	_alpha(1./0.),
	_integral_r2_g2_dr(0.),
	_norm_r2_g2_dr(0.),
	_integral_r2_g_dr(0.),
	_norm_r2_g_dr(0.) {
	;
}

double RadialGaussian::at(double r) {
	double p = _alpha*(r-_r0)*(r-_r0);
	if (p < 40) return _norm_r2_g2_dr * exp(-p);
	else return 0.0;
}

// ======================
// SphericalGaussian
// ======================

SphericalGaussian::SphericalGaussian(vec r0, double sigma) :
	_r0(r0), _sigma(sigma), _alpha(1./(2*sigma*sigma)) {
	_norm_g_dV = pow(_alpha/M_PI, 1.5);
}

// ======================
// ModSphBessel1stKind
// ======================

std::vector<double> ModifiedSphericalBessel1stKind::eval(int degree, double r) {
	std::vector<double> il;
	if (r < RADZERO) {
		il.push_back(1.);
		il.push_back(0.);
	}
	else {
		il.push_back(sinh(r)/r);
		il.push_back(cosh(r)/r - sinh(r)/(r*r));
	}
	for (int l = 2; l <= degree; ++l) {
		if (r < RADZERO) {
			il.push_back(0.);
		}
		else {
			if (il[l-1] < SPHZERO) il.push_back(0.);
			else il.push_back( il[l-2] - (2*(l-1)+1)/r*il[l-1] );
		}
	}
	return il;
}

} /* CLOSE NAMESPACE */
