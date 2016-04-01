#include <math.h>
#include <boost/math/special_functions/erf.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

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
    _Rc = options.get<double>("radialcutoff.Rc");
    _integration_steps = options.get<int>("radialbasis.integration_steps");
    _mode = options.get<std::string>("radialbasis.mode");
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

    if (_mode == "equispaced") {
		double dr = _Rc/(_N-1);
		for (int i = 0; i < _N; ++i) {
			double r = i*dr;
			double sigma = _sigma;
			basis_fct_t *new_fct = new basis_fct_t(r, sigma);
			_basis.push_back(new_fct);
		}
    }
    else if (_mode == "adaptive") {
        //double delta = 0.5;
        int L = 6;
        double r = 0.;
        double sigma = 0.;
        double sigma_0 = 0.5;
        double sigma_stride_factor = 0.5;
//        while (r < _Rc) {
//            sigma = sqrt(4./(2*L+1))*(r+delta);
//            basis_fct_t *new_fct = new basis_fct_t(r, sigma);
//			_basis.push_back(new_fct);
//			r = r + sigma;
//        }
        for (int i = 0; i < _N; ++i) {
        	//sigma = sqrt(4./(2*L+1))*(r+delta);
        	sigma = sqrt(4./(2*L+1)*r*r + sigma_0*sigma_0);
        	std::cout << r << " " << sigma << std::endl;
        	r = r + sigma_stride_factor*sigma;
        }
        double scale = 1.; //_Rc/(r-sigma);
        r = 0;
        for (int i = 0; i < _N; ++i) {
        	sigma = sqrt(4./(2*L+1)*r*r + sigma_0*sigma_0);
        	basis_fct_t *new_fct = new basis_fct_t(scale*r, sigma);
			_basis.push_back(new_fct);
			std::cout << r << " " << sigma << std::endl;
			r = r+sigma_stride_factor*sigma;
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
    // REPORT
	GLOG() << "Radial basis overlap matrix" << std::endl;
	for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
		for (jt = _basis.begin(), j = 0; jt != _basis.end(); ++jt, ++j) {
			 GLOG() << boost::format("%1$+1.4e") % _Sij(i,j) << " " << std::flush;
		}
		GLOG() << std::endl;
	}

    // ORTHONORMALIZATION VIA CHOLESKY DECOMPOSITION
    _Uij = _Sij;
    soap::linalg::linalg_cholesky_decompose(_Uij);
    // REPORT
    GLOG() << "Radial basis Cholesky decomposition" << std::endl;
	for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i) {
		for (jt = _basis.begin(), j = 0; jt != _basis.end(); ++jt, ++j) {
			 GLOG() << boost::format("%1$+1.4e") % _Uij(i,j) << " " << std::flush;
		}
		GLOG() << std::endl;
	}

    // ZERO UPPER DIAGONAL OF U
    for (it = _basis.begin(), i = 0; it != _basis.end(); ++it, ++i)
		for (jt = it+1, j = i+1; jt != _basis.end(); ++jt, ++j)
			 _Uij(i,j) = 0.0;
    _Tij = _Uij;
    soap::linalg::linalg_invert(_Uij, _Tij);
    // REPORT
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

struct ModifiedSphericalBessel1stKind
{
	static std::vector<double> eval(int degree, double r) {
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

	static constexpr double RADZERO = 1e-10;
	static constexpr double SPHZERO = 1e-4;
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
//		std::cout << "sigma > 0." << std::endl;

//		int degree_sph_in = save_here.size2();
//		std::vector<double> sph_il = ModifiedSphericalBessel1stKind::eval(degree_sph_in, 1e-4);
//		for (int l = 0; l != save_here.size2(); ++l) {
//			std::cout << "l=" << l << " il=" << sph_il[l] << std::endl;
//		}
//
//		throw soap::base::NotImplemented("...");

		// Particle properties
		double ai = 1./(2*particle_sigma*particle_sigma);
		double ri = r;
		SphericalGaussian gi_sph(vec(0,0,0), particle_sigma); // <- position should not matter, as only normalization used here
		double norm_g_dV_sph_i = gi_sph._norm_g_dV;

		int k = 0;
		basis_it_t it;
		for (it = _basis.begin(), k = 0; it != _basis.end(); ++it, ++k) {
			// Radial Gaussian properties
			double ak = (*it)->_alpha;
			double rk = (*it)->_r0;
			double norm_r2_g2_dr_rad_k = (*it)->_norm_r2_g2_dr;
			// Combined properties
			double beta_ik = ai+ak;
			double rho_ik = ak*rk/beta_ik;

			double sigma_ik = sqrt(0.5/beta_ik);
			RadialGaussian gik_rad(rho_ik, sigma_ik);
			double norm_r2_g_dr_rad_ik = gik_rad._norm_r2_g_dr;

			double prefac =
			    4*M_PI *
				norm_r2_g2_dr_rad_k*norm_g_dV_sph_i/norm_r2_g_dr_rad_ik *
				exp(-ai*ri*ri) *
				exp(-ak*rk*rk*(1-ak/beta_ik));

//			// DELTA APPROXIMATION
//			std::cout << "r " << r;
//			std::cout << " beta_ik " << beta_ik;
//			std::cout << " rho_ik " << rho_ik;
//			std::cout << std::endl;
//			double bessel_arg_ik = 2*ai*ri*rho_ik;
//			int degree_sph = save_here.size2(); // <- L+1
//			std::vector<double> sph_il = ModifiedSphericalBessel1stKind::eval(degree_sph, bessel_arg_ik);
//            for (int l = 0; l != degree_sph; ++l) {
//            	save_here(k, l) = prefac*sph_il[l];
//            }
//            std::cout << std::endl;
//            std::cout << "DELTA" << std::endl;
//            std::cout << "k = " << k;
//			for (int l = 0; l != save_here.size2(); ++l) {
//				std::cout << " " << save_here(k, l);
//			}
//			std::cout << std::endl;


			// NUMERICAL INTEGRATION
			// ZERO COEFFS
			for (int l = 0; l != save_here.size2(); ++l) {
				save_here(k, l) = 0.0;
			}

//			// INTEGRATE
//			double delta_r = 0.01;
//			double r_min = 0.0;
//			double r_max = 10.;
//			int steps = int((r_max-r_min)/delta_r)+1;
//			std::cout << "r " << r;
//			std::cout << " steps " << steps;
//			std::cout << " beta_ik " << beta_ik;
//			std::cout << " rho_ik " << rho_ik;
//			std::cout << std::endl;
//
//			for (int s = 0; s <= steps; ++s) {
//				// Current radius
//				double r_step = r_min + s*delta_r;
//				// Calculate ModSphBessels
//				double arg = 2*ai*ri*r_step;
//				std::vector<double> sph_il = ModifiedSphericalBessel1stKind::eval(save_here.size2(), arg);
//				// Add increments
//				for (int l = 0; l != save_here.size2(); ++l) {
//					save_here(k,l) +=
//						prefac*
//						delta_r*r_step*r_step*
//						sph_il[l]*
//						exp(-beta_ik*(r_step-rho_ik)*(r_step-rho_ik))*norm_r2_g_dr_rad_ik;
//				}
//			}

			// SIMPSON'S RULE
			double r_min = rho_ik - 4*sigma_ik;
			double r_max = rho_ik + 4*sigma_ik;
			if (r_min < 0.) {
				r_max -= r_min;
				r_min = 0.;
			}
			int n_steps = this->_integration_steps;
			double delta_r_step = (r_max-r_min)/(n_steps-1);
			int n_sample = 2*n_steps+1;
			double delta_r_sample = 0.5*delta_r_step;

			// Compute samples for all l's
			// For each l, store integrand at r_sample
			ub::matrix<double> integrand_l_at_r = ub::zero_matrix<double>(save_here.size2(), n_sample);
			for (int s = 0; s < n_sample; ++s) {
				// f0 f1 f2 f3 ....  f-3 f-2 f-1
				// |-----||----||----||-------|
				double r_sample = r_min - delta_r_sample + s*delta_r_sample;
				// ... Generate Bessels
				std::vector<double> sph_il = ModifiedSphericalBessel1stKind::eval(save_here.size2(), 2*ai*ri*r_sample);
				// ... Compute & store integrands
				for (int l = 0; l != save_here.size2(); ++l) {
					integrand_l_at_r(l, s) =
					    prefac*
						r_sample*r_sample*
						sph_il[l]*
						exp(-beta_ik*(r_sample-rho_ik)*(r_sample-rho_ik))*norm_r2_g_dr_rad_ik;
				}
			}
			// Apply Simpson's rule
			for (int s = 0; s < n_steps; ++s) {
				for (int l = 0; l != save_here.size2(); ++l) {
					save_here(k,l) +=
						delta_r_step/6.*(
							integrand_l_at_r(l, 2*s)+
							4*integrand_l_at_r(l, 2*s+1)+
							integrand_l_at_r(l, 2*s+2)
						);
				}
			}

//			std::cout << "NUMERICAL" << std::endl;
//			std::cout << "k = " << k;
//			for (int l = 0; l != save_here.size2(); ++l) {
//				std::cout << " " << save_here(k, l);
//			}
//			std::cout << std::endl;


		}
		save_here = ub::prod(_Tij, save_here);
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

BOOST_CLASS_EXPORT_IMPLEMENT(soap::RadialBasisGaussian);

