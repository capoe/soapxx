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
        _din.resize(_degree+1, 0.);
        if (r < RADZERO) {
            _din[1] = 1./3.;
            //_din.push_back(0.0);
            //_din.push_back(1./3.);
            //for (int n = 2; n <= _degree; ++n) {
            //    _din.push_back(0.);
            //}
        }
        else {
            _din[0] = _in[1];
            for (int n = 1; n <= _degree; ++n) {
                _din[n] = _in[n-1] - (n+1.)/r*_in[n];
            }
            //_din.push_back( _in[1] );
            //for (int n = 1; n <= _degree; ++n) {
            //    _din.push_back( _in[n-1] - (n+1.)/r*_in[n] );
            //}
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
        // This should have been checked before
        // All gradients set to zero
        throw soap::base::APIError("<GradSphericalYlm::eval> R < RADZERO");
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

const int FACTORIAL_CACHE_SIZE = 19;
const long int FACTORIAL_CACHE[] = {
  1,
  1, 2, 6,
  24, 120, 720,
  5040, 40320, 362880,
  3628800, 39916800, 479001600,
  6227020800, 87178291200, 1307674368000,
  20922789888000, 355687428096000, 6402373705728000};

long int factorial(int x) {
    if (x < FACTORIAL_CACHE_SIZE) {
        return FACTORIAL_CACHE[x];
    }
    else {
        assert(false && "Computing factorial from scratch - not desirable");
        long int s = 1.0;
        for (int n = 2; n <= x; n++) {
          s *= n;
        }
        return s;
    }
}

const int FACTORIAL2_CACHE_SIZE = 19;
const long int FACTORIAL2_CACHE[] = {
    1,
    1, 2, 3,
    8, 15, 48,
    105, 384, 945,
    3840, 10395, 46080,
    135135, 645120, 2027025,
    10321920, 34459425, 185794560};

long int factorial2(int x) {
    if (x < 0) {
        assert(x == -1);
        return 1;
    }
    else if (x < FACTORIAL2_CACHE_SIZE) {
        return FACTORIAL2_CACHE[x];
    }
    else {
        assert(false && "Computing factorial from scratch - not desirable");
        long int s = 1;
        for (int n = x; n > 0; n -= 2) {
            s *= n;
        }
        return s;
    }
}

void calculate_solidharm_Rlm(
    vec d,
    double r,
    int L,
    std::vector<std::complex<double>> &rlm) {
    // Initialise
    rlm.clear();
    rlm.resize((L+1)*(L+1), 0.0);
    // Treat r=0
    if (r < 1e-10) { // TODO Define SPACE-QUANTUM
        //throw soap::base::APIError("Rlm(r=0) disabled by design: Handle r=0-case manually.");
        rlm[0] = 1.;
        return;
    }
    // Proceed with r != 0: Compute Associated Legendre Polynomials
    std::vector<double> plm;
	double theta = acos(d.getZ());
	double phi = atan2(d.getY(), d.getX());
	if (phi < 0.) phi += 2*M_PI; // <- Shift [-pi, -0] to [pi, 2*pi]
	calculate_legendre_plm(L, d.getZ(), plm);
    // Add radial component, phase factors, normalization
    for (int l = 0; l <= L; ++l) {
        for (int m = 0; m <= l; ++m) {
            rlm[l*l+l+m] =
                  pow(-1, l-m)
                * pow(r, l)
                / factorial(l+m)
                * std::exp(std::complex<double>(0.,m*phi))
                * plm[l*(l+1)/2+m];
        }
        for (int m = -l; m < 0; ++m) {
            rlm[l*l+l+m] = pow(-1, m)*std::conj(rlm[l*l+l-m]);
        }
    }
    return;
}

void calculate_solidharm_rlm_ilm(
        vec d,
        double r,
        int L,
        std::vector<std::complex<double>> &rlm,
        std::vector<std::complex<double>> &ilm) {

    std::vector<double> plm;
    double theta = 0.0;
    double phi = 0.0;

    // Compute Associated Legendre Polynomials
    if (r < 1e-10) { // TODO Define SPACE-QUANTUM
        assert(false && "Should not calculate irregular spherical harmonics with r=0");
        plm.resize(L*(L+1)/2+L+1, 0.0);
        plm[0] = boost::math::legendre_p(0, 0, 0);
    }
    else {
		theta = acos(d.getZ());
		phi = atan2(d.getY(), d.getX());
		if (phi < 0.) phi += 2*M_PI; // <- Shift [-pi, -0] to [pi, 2*pi]
		calculate_legendre_plm(L, d.getZ(), plm);
    }

    std::cout << "phi = " << phi << std::endl;

    rlm.clear();
    rlm.resize((L+1)*(L+1), 0.0);
    for (int l = 0; l <= L; ++l) {
        for (int m = 0; m <= l; ++m) {
            rlm[l*l+l+m] =
                  pow(-1, l-m)
                * pow(r, l)
                / factorial(l+m)
                * std::exp(std::complex<double>(0.,m*phi))
                * plm[l*(l+1)/2+m];
        }
        for (int m = -l; m < 0; ++m) {
            rlm[l*l+l+m] = pow(-1, m)*std::conj(rlm[l*l+l-m]);
        }
    }

    ilm.clear();
    ilm.resize((L+1)*(L+1), 0.0);
    for (int l = 0; l <= L; ++l) {
        for (int m = 0; m <= l; ++m) {
            ilm[l*l+l+m] =
                  pow(-1, l-m)
                * pow(r, -l-1)
                * factorial(l-m)
                * std::exp(std::complex<double>(0.,m*phi))
                * plm[l*(l+1)/2+m];
        }
        for (int m = -l; m < 0; ++m) {
            ilm[l*l+l+m] = pow(-1, m)*std::conj(ilm[l*l+l-m]);
        }
    }

    return;
}

void calculate_legendre_plm(int L, double x, std::vector<double> &plm_out) {
    // See 'Numerical Recipes' - Chapter on Special Functions
    plm_out.clear();
    plm_out.resize(L*(L+1)/2+L+1, 0.0);
    // Pll / Pmm
    for (int l = 0; l <= L; ++l) {
        plm_out[l*(l+1)/2+l] = boost::math::legendre_p(l, l, x);
        //GLOG() << l << ":" << l << std::endl;
    }
    // Pmm => Pm+1m
    for (int l = 0; l < L; ++l) {
        int m = l;
        int ll = l+1;
        plm_out[ll*(ll+1)/2+l] = x*(2*m+1)*plm_out[l*(l+1)/2+m];
        //GLOG() << ll << ":" << l << std::endl;
    }
    // Pl-1m,Pl-2m => Plm
    for (int m = 0; m <= L; ++m) {
        for (int l2 = m+2; l2 <= L; ++l2) {
            int l0 = l2-2;
            int l1 = l2-1;
            int lm0 = l0*(l0+1)/2+m;
            int lm1 = l1*(l1+1)/2+m;
            int lm2 = l2*(l2+1)/2+m;
            //GLOG() << l2 << ":" << m << " from " << lm1 << "," << lm0 << std::endl;
            plm_out[lm2] = (x*(2*l2-1)*plm_out[lm1] - (l2+m-1)*plm_out[lm0])/(l2-m);
        }
    }
}

} /* CLOSE NAMESPACE */
