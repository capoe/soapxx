#include <math.h>
#include <boost/math/special_functions/erf.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "base/exceptions.hpp"
#include "linalg/operations.hpp"
#include "globals.hpp"
#include "radialbasis.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

// ======================
// RadialBasis BASE CLASS
// ======================

void RadialBasis::configure(Options &options) {
    _N = options.get<int>("radialbasis.N");
    _Rc = options.get<double>("radialbasis.Rc");
}

RadialCoefficients RadialBasis::computeCoefficients(double r) {
	RadialCoefficients coeffs(this->_N);
	throw soap::base::NotImplemented("RadialBasis::computeCoefficients");
	return coeffs;
}

void RadialBasis::computeCoefficients(double r, double particle_sigma, radcoeff_t &save_here) {
	throw soap::base::NotImplemented("RadialBasis::computeCoefficients");
	return;
}

RadialCoefficients RadialBasis::computeCoefficientsAllZero() {
	RadialCoefficients coeffs(this->_N);
    for (int i = 0; i < _N; ++i) coeffs(i) = 0.0;
	return coeffs;
}


// ======================
// RadialBasisGaussian
// ======================

void RadialBasisGaussian::configure(Options &options) {
	RadialBasis::configure(options);
    _sigma = options.get<double>("radialbasis.sigma");

    // CREATE & STORE EQUISPACED RADIAL GAUSSIANS
    this->clear();

    std::string mode = "adaptive";
    mode = "equispaced";
    if (mode == "equispaced") {
		double dr = _Rc/(_N-1);
		for (int i = 0; i < _N; ++i) {
			double r = i*dr;
			double sigma = _sigma;
			basis_fct_t *new_fct = new basis_fct_t(r, sigma);
			_basis.push_back(new_fct);
		}
    }
    else if (mode == "adaptive") {
        double delta = 0.5;
        int L = 6;
        double r = 0.;
        double sigma = 0.;
        while (r < _Rc) {
            sigma = sqrt(4./(2*L+1))*(r+delta);
            basis_fct_t *new_fct = new basis_fct_t(r, sigma);
			_basis.push_back(new_fct);
			r = r + sigma;
        }
    }
    else {
    	throw std::runtime_error("Not implemented.");
    }
    // SUMMARIZE
    GLOG() << "Created " << _N << " radial Gaussians at r = { ";
    for (basis_it_t bit = _basis.begin(); bit != _basis.end(); ++bit) {
        GLOG() << (*bit)->_r0 << " (" << (*bit)->_sigma << ") ";
    }
    GLOG() << "}" << std::endl;
    // COMPUTE OVERLAP MATRIX s_{ij} = \int g_i(r) g_j(r) r^2 dr
    _Sij = ub::matrix<double>(_N, _N);
    basis_it_t it;
    basis_it_t jt;
    int i;
    int j;
    for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
    	for (jt = _basis.begin(), j = 0; jt != _basis.end(); ++jt, ++j) {
            double a = (*it)->_alpha;
            double b = (*jt)->_alpha;
            double r0 = (*it)->_r0;
            double r1 = (*jt)->_r0;
            double w = a + b;
            double W0 = a*r0 + b*r1;
            double s =
				1./(4.*pow(w, 2.5))*exp(-a*r0*r0 -b*r1*r1)*(
					2*sqrt(w)*W0 +
					sqrt(M_PI)*exp(W0*W0/w)*(w+2*W0*W0)*(
						1 - boost::math::erf<double>(-W0/sqrt(w))
					)
				);
			s *= (*it)->_norm_r2_g2_dr*(*jt)->_norm_r2_g2_dr;
			_Sij(i,j) = s;
    	}
    }
    // ORTHONORMALIZATION VIA CHOLESKY DECOMPOSITION
    _Uij = _Sij;
    soap::linalg::linalg_cholesky_decompose(_Uij);
    // ZERO UPPER DIAGONAL OF U
    for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i)
		for (jt = it+1, j = i+1; jt != _basis.end(); ++jt, ++j)
			 _Uij(i,j) = 0.0;
    _Tij = _Uij;
    soap::linalg::linalg_invert(_Uij, _Tij);
    // REPORT
	GLOG() << "Radial basis overlap matrix" << std::endl;
	for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
		for (jt = _basis.begin(), j = 0; jt != _basis.end(); ++jt, ++j) {
			 GLOG() << boost::format("%1$+1.4e") % _Sij(i,j) << " " << std::flush;
		}
		GLOG() << std::endl;
	}
    GLOG() << "Radial basis Cholesky decomposition" << std::endl;
    for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
		for (jt = _basis.begin(), j = 0; jt != _basis.end(); ++jt, ++j) {
			 GLOG() << boost::format("%1$+1.4e") % _Uij(i,j) << " " << std::flush;
		}
		GLOG() << std::endl;
	}
    GLOG() << "Radial basis transformation matrix" << std::endl;
	for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
		for (jt = _basis.begin(), j = 0; jt != _basis.end(); ++jt, ++j) {
			 GLOG() << boost::format("%1$+1.4e") % _Tij(i,j) << " " << std::flush;
		}
		GLOG() << std::endl;
	}
}

RadialCoefficients RadialBasisGaussian::computeCoefficients(double r) {

	ub::vector<double> coeffs_raw(_N);
	basis_it_t it;
	int i;
    for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
        coeffs_raw(i) = (*it)->at(r);
    }

    coeffs_raw = ub::prod(_Tij, coeffs_raw);

    RadialCoefficients coeffs(this->_N);
    for (int n = 0; n < _N; ++n) {
    	coeffs.set(n, coeffs_raw(n));
    }

	return coeffs;
}

struct SphericalGaussian
{
	SphericalGaussian(vec r0, double sigma) :
		_r0(r0), _sigma(sigma), _alpha(1./(2*sigma*sigma)) {
		_norm_g_dV = pow(_alpha/M_PI, 1.5);
	}

	vec _r0;
	double _sigma;
	double _alpha;
	double _norm_g_dV;
};


void RadialBasisGaussian::computeCoefficients(double r, double particle_sigma, radcoeff_t &save_here) {
	// Delta-type expansion =>
	// Second (l) dimension of <save_here> and <particle_sigma> ignored here
	if (particle_sigma < RadialBasis::RADZERO) {
		basis_it_t it;
		int n = 0;
		for (it = _basis.begin(), n = 0; it != _basis.end(); ++it, ++n) {
			double gn_at_r = (*it)->at(r);
			for (int l = 0; l != save_here.size2(); ++l) {
				save_here(n, l) = gn_at_r;
			}
		}

		save_here = ub::prod(_Tij, save_here);
	}
	else {
		throw soap::base::NotImplemented("...");

		// Particle properties
		double ai = 1./(2*particle_sigma*particle_sigma);
		double ri = r;

		basis_it_t it;
		int k = 0;
		for (it = _basis.begin(), k = 0; it != _basis.end(); ++it, ++k) {
			// Radial Gaussian properties
			double ak = (*it)->_alpha;
			double rk = (*it)->_r0;
			// Combined properties
			double beta_ik = ai+ak;
			double rho_ik = ak*rk/beta_ik;
			double bessel_arg_ik = 2*ai*ri*rho_ik;


		}
	}
    return;
}


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

double RadialGaussian::at(double r) {
	double p = _alpha*(r-_r0)*(r-_r0);
	if (p < 40) return _norm_r2_g2_dr * exp(-p);
	else return 0.0;
}


// ======================
// RadialBasisFactory
// ======================

void RadialBasisFactory::registerAll(void) {
	RadialBasisOutlet().Register<RadialBasisGaussian>("gaussian");
	RadialBasisOutlet().Register<RadialBasisLegendre>("legendre");
}

}

