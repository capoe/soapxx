#include "radialbasis.hpp"
#include "globals.hpp"
#include "linalg/operations.hpp"
#include <math.h>
#include <boost/math/special_functions/erf.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace soap {

namespace ub = boost::numeric::ublas;


void RadialBasis::configure(Options &options) {
    _N = options.get<int>("radialbasis.N");
    _Rc = options.get<double>("radialbasis.Rc");
}


void RadialBasisGaussian::configure(Options &options) {
	RadialBasis::configure(options);
    _sigma = options.get<double>("radialbasis.sigma");

    // CREATE & STORE EQUISPACED RADIAL GAUSSIANS
    this->clear();
    double dr = _Rc/(_N-1);
    for (int i = 0; i < _N; ++i) {
        double r = i*dr;
        double sigma = _sigma;
        basis_fct_t *new_fct = new basis_fct_t(r, sigma);
        _basis.push_back(new_fct);
    }
    // SUMMARIZE
    GLOG() << "Created " << _N << " radial Gaussians at r = { ";
    for (basis_it_t bit = _basis.begin(); bit != _basis.end(); ++bit) {
        GLOG() << (*bit)->_r0 << " ";
    }
    GLOG() << "}" << std::endl;
    // COMPUTE OVERLAP MATRIX
    _Sij = ub::matrix<double>(_N, _N);
    basis_it_t it;
    basis_it_t jt;
    int i;
    int j;
    for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
    	for (jt = _basis.begin(), j = 0; jt != _basis.end(); ++jt, ++j) {
    		/*
    		double a = (*it)->_alpha;
    		double a32 = pow(a, 1.5);
    		double b = (*it)->_r0;
    		double c = (*jt)->_r0;
    		double pre = 4.*M_PI/(16.*a32);
			double s = pre*exp(-a*(b*b+c*c)) * (
			  sqrt(2.*M_PI)*(1.+a*(b+c)*(b+c))*exp(0.5*a*(b+c)*(b+c)) * \
				 (1. - boost::math::erf<double>(-1.*sqrt(0.5*a)*(b+c)))
			  + 2.*sqrt(a)*(b+c)
			);
			*/
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

			s *= (*it)->_norm_dV*(*jt)->_norm_dV;
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


RadialGaussian::RadialGaussian(double r0, double sigma)
: _r0(r0),
  _sigma(sigma),
  _alpha(1./(2.*sigma*sigma)) {
	// COMPUTE NORMALIZATION
    /*
	double alpha2 = 2*_alpha;
    _integral_4_pi_r2_g2_dr =
        pow(M_PI/alpha2, 1.5) * (1.+2.*alpha2*_r0*_r0)*(1.-boost::math::erf<double>(-sqrt(alpha2)*_r0))
      + pow(M_PI/alpha2, 1.5) * 2*sqrt(alpha2/M_PI)*_r0*exp(-alpha2*_r0*_r0);
    _norm_dV = 1./std::sqrt(_integral_4_pi_r2_g2_dr);
    */
    double w = 2*_alpha;
    double W0 = 2*_alpha*_r0;
    _integral_r2_g2_dr =
        1./(4.*pow(w, 2.5))*exp(-2*_alpha*_r0*_r0)*(
            2*sqrt(w)*W0 +
			sqrt(M_PI)*exp(W0*W0/w)*(w+2*W0*W0)*(
			    1 - boost::math::erf<double>(-W0/sqrt(w))
			)
        );
    _norm_dV = 1./sqrt(_integral_r2_g2_dr);
}

/*
self.r0 = r0
self.sigma = sigma
self.alpha = 1./(2*sigma**2)
# Normalization
alpha2 = 2*self.alpha
self.integral_4_pi_r2_g2_dr = (np.pi/alpha2)**1.5*(1.+2.*alpha2*self.r0**2)*(1.-scipy.special.erf(-alpha2**0.5*self.r0)) + (np.pi/alpha2)**1.5*2*(alpha2/np.pi)**0.5*self.r0*np.exp(-alpha2*self.r0**2)
self.norm_dV = self.integral_4_pi_r2_g2_dr**(-0.5)
return
def EvaluateExp(self, r):
return np.exp(-(r-self.r0)**2/(2.*self.sigma**2))
def __call__(self, r):
return self.EvaluateExp(r)*self.norm_dV
*/



void RadialBasisFactory::registerAll(void) {
	RadialBasisOutlet().Register<RadialBasisGaussian>("gaussian");
	RadialBasisOutlet().Register<RadialBasisLegendre>("legendre");
}

}

