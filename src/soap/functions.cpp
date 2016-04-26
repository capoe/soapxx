#include "soap/functions.hpp"
#include "soap/base/exceptions.hpp"

#include <math.h>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

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

ModifiedSphericalBessel1stKind::ModifiedSphericalBessel1stKind(int degree) :
    _degree(degree) {
    _in.reserve(degree);
    _din.reserve(degree);
}

void ModifiedSphericalBessel1stKind::evaluate(double r, bool differentiate) {
    _in.clear();
    _din.clear();

    _in = ModifiedSphericalBessel1stKind::eval(_degree, r);

    if (differentiate) {
        if (r < RADZERO) {
            _din.push_back(0.0);
            _din.push_back(1./3.);
            for (int n = 1; n <= _degree; ++n) {
                _din.push_back(0.);
            }
        }
        else {
            _din.push_back( _in[1] );
            for (int n = 1; n <= _degree; ++n) {
                _din.push_back( _in[n-1] - (n+1.)/r*_in[n] );
            }
        }
    }

    return;
}

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
			il.push_back( il[l-2] - (2*(l-1)+1)/r*il[l-1] );
		}
	}
	return il;
}

// ==============================
// GradSphericalYlm \Nabla Y_{lm}
// ==============================

std::vector<std::complex<double> > GradSphericalYlm::eval(int l, int m, vec &r) {

    std::vector<std::complex<double> > dylm;
    dylm.push_back(std::complex<double>(0.,0.));
    dylm.push_back(std::complex<double>(0.,0.));
    dylm.push_back(std::complex<double>(0.,0.));

    // TODO WHAT IF RADIUS IS ZERO?
    double R = soap::linalg::abs(r);
    if (R < RADZERO) {
        throw soap::base::NotImplemented("<GradSphericalYlm::eval> R < RADZERO");
    }
    vec d = r / R;
    double x = r.getX();
    double y = r.getY();
    double z = r.getZ();

    // COMPUTE YLM
    double theta = acos(d.getZ());
    double phi = atan2(d.getY(), d.getX());
    // Shift [-pi, -0] to [pi, 2*pi]
    if (phi < 0.) phi += 2*M_PI;
    cmplx ylm = boost::math::spherical_harmonic<double,double>(l, m, theta, phi);

    // COMPUTE DYLM
    int p, q, s;
    for (p = 0; p <= l; ++p) {
         q = p - m;
         s = l - p - q;

         //std::cout << p << "_" << q << "_" << s << std::endl;
         //std::cout << "DYLM" << dylm[0] << dylm[1] << dylm[2] << std::endl;

         double fact_p;
         double fact_q;
         double fact_s;
         if ((p >= 0)) {
             fact_p = factorial(p);
         }
         if (q >= 0) {
              fact_q = factorial(q);
         }
         if ((s >= 0)) {
              fact_s = factorial(s);
         }

         if ((p >= 1) && (q >= 0) && (s >= 0)) {
             double fact_p_1 = factorial(p-1);
            dylm[0] = dylm[0]
                - (( pow_nnan(cmplx(-0.5 * x, -0.5 * y),(p - 1)) )
                * ( pow_nnan(cmplx(0.5 * x, -0.5 * y),q) )
                * ( pow(z,s) )
                * 0.5
                / (  fact_p_1*fact_q*fact_s  ));
            dylm[1] = dylm[1]
                - (( pow_nnan(cmplx(-0.5 * x, -0.5 * y),(p - 1)) )
                * ( pow_nnan(cmplx(0.5 * x, -0.5 * y),q) )
                * ( pow(z,s) )
                * 0.5 * cmplx(0.0, 1.0)
                / (  fact_p_1*fact_q*fact_s  ));
         }

         if ((p >= 0) && (q >= 1) && (s >= 0)) {
             double fact_q_1 = factorial(q-1);
             dylm[0] = dylm[0]
                + (( pow_nnan(cmplx(-0.5 * x, -0.5 * y),p) )
                * ( pow_nnan(cmplx(0.5 * x, -0.5 * y),(q - 1)) )
                * ( pow(z,s) )
                * 0.5
                / ( fact_p*fact_q_1*fact_s ));
             dylm[1] = dylm[1]
                - (( pow_nnan(cmplx(-0.5 * x, -0.5 * y),p) )
                * ( pow_nnan(cmplx(0.5 * x, -0.5 * y),(q - 1)) )
                * ( pow(z,s) )
                * 0.5 * cmplx(0.0, 1.0)
                / ( fact_p*fact_q_1*fact_s ));
         }

         if ((p >= 0) && (q >= 0) && (s >= 1)) {
             double fact_s_1 = factorial(s-1);
             dylm[2] = dylm[2]
                + (( pow_nnan(cmplx(-0.5 * x, -0.5 * y),p) )
                * ( pow_nnan(cmplx(0.5 * x, -0.5 * y),q) )
                * ( pow(z,(s - 1)) )
                / ( fact_p*fact_q*fact_s_1 ));
         }
         //std::cout << "DYLM-OUT" << dylm[0] << dylm[1] << dylm[2] << std::endl;
    }

    double prefac = sqrt(factorial(l + m) * factorial(l - m) * ((2.0 * l) + 1) / (4.0 * M_PI)) * ( pow(R*R,(-0.5 * l)) );
    dylm[0] = dylm[0]*prefac;
    dylm[1] = dylm[1]*prefac;
    dylm[2] = dylm[2]*prefac;

    dylm[0] = dylm[0] - (l*x*ylm/(R*R));
    dylm[1] = dylm[1] - (l*y*ylm/(R*R));
    dylm[2] = dylm[2] - (l*z*ylm/(R*R));

    return dylm;
}

std::complex<double> pow_nnan(std::complex<double> z, double a) {
    if (a == 0) {
        return std::complex<double>(1.,0.);
    }
    else {
        return pow(z,a);
    }
}

int factorial(int n) {
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

} /* CLOSE NAMESPACE */
