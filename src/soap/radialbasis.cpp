#include <math.h>
#include <boost/math/special_functions/erf.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "soap/base/exceptions.hpp"
#include "soap/linalg/operations.hpp"
#include "soap/globals.hpp"
#include "soap/radialbasis.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

// ======================
// RadialBasis BASE CLASS
// ======================

void RadialBasis::configure(Options &options) {
    _N = options.get<int>("radialbasis.N");
    _Rc = options.get<double>("radialcutoff.Rc"); // <- !! Modified below for adaptive basis sets !!
    _integration_steps = options.get<int>("radialbasis.integration_steps");
    _mode = options.get<std::string>("radialbasis.mode");
}

void RadialBasis::computeCoefficients(
        vec d,
        double r,
        double particle_sigma,
        radcoeff_t &Gnl,
        radcoeff_t *dGnl_dx,
        radcoeff_t *dGnl_dy,
        radcoeff_t *dGnl_dz) {
	throw soap::base::NotImplemented("RadialBasis::computeCoefficients");
	return;
}

// ======================
// RadialBasisGaussian
// ======================

RadialBasisGaussian::RadialBasisGaussian() {
	 _type = "gaussian";
	 _is_ortho = false;
	 _sigma = 0.5;
}

void RadialBasisGaussian::clear() {
	for (basis_it_t bit=_basis.begin(); bit!=_basis.end(); ++bit) {
		delete *bit;
	}
	_basis.clear();
}

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
        int L = options.get<int>("angularbasis.L");
        double r = 0.;
        double sigma = 0.;
        double sigma_0 = _sigma;
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
        	//std::cout << r << " " << sigma << std::endl;
        	r = r + sigma_stride_factor*sigma;
        }
        double scale = 1.; //_Rc/(r-sigma);
        r = 0;
        for (int i = 0; i < _N; ++i) {
        	sigma = sqrt(4./(2*L+1)*r*r + sigma_0*sigma_0);
        	basis_fct_t *new_fct = new basis_fct_t(scale*r, sigma);
			_basis.push_back(new_fct);
			//std::cout << r << " " << sigma << std::endl;
			r = r+sigma_stride_factor*sigma;
        }
        _Rc = r - sigma_stride_factor*sigma;
        options.set("radialcutoff.Rc", _Rc);
        options.set("radialcutoff.Rc_heaviside", _Rc+1.5*sigma);
        GLOG() << "Adjusted radial cutoff to " << _Rc
        	<< " based on sigma_0 = " << sigma_0 << ", L = " << L << ", stride = " << sigma_stride_factor << std::endl;
    }
    else if (_mode == "incremental") {
        throw std::runtime_error("Not implemented.");
    }
    else {
    	throw std::runtime_error("Not implemented.");
    }
    // SUMMARIZE
    GLOG() << "Created " << _N << " radial Gaussians at:" << std::endl;
    for (basis_it_t bit = _basis.begin(); bit != _basis.end(); ++bit) {
        GLOG() << boost::format(" r = %1$+1.7e") % (*bit)->_r0
            << boost::format(" (sigma = %1$+1.7e") % (*bit)->_sigma << ") "
            << std::endl;
    }

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


// For each l: integral S r^2 dr i_l(2*ai*ri*r) exp(-beta_ik*(r-rho_ik)^2)
void compute_integrals_il_expik_r2_dr(
        double ai,
        double ri,
        double beta_ik,
        double rho_ik,
        int L_plus_1,
        int n_steps,
        std::vector<double> *integrals,
        std::vector<double> *integrals_derivative) {

    bool gradients = (integrals_derivative != NULL) ? true : false;

    // Sanity checks
    assert(integrals->size() == L_plus_1);
    if (gradients) assert(integrals->size() == L_plus_1);

    // Sample coordinates along r-axis
    double sigma_ik = sqrt(0.5/beta_ik);
    double r_min = rho_ik - 4*sigma_ik;
    double r_max = rho_ik + 4*sigma_ik;
    if (r_min < 0.) {
        r_max -= r_min;
        r_min = 0.;
    }
    double delta_r_step = (r_max-r_min)/(n_steps-1);
    int n_sample = 2*n_steps+1;
    double delta_r_sample = 0.5*delta_r_step;

    ModifiedSphericalBessel1stKind mosbest(L_plus_1-1);

    if (gradients) {
        // Compute samples for all l's ...
        ub::matrix<double> integrand_l_at_r = ub::zero_matrix<double>(L_plus_1, n_sample);
        ub::matrix<double> integrand_derivative_l_at_r = ub::zero_matrix<double>(L_plus_1, n_sample);
        for (int s = 0; s < n_sample; ++s) {
            double r_sample = r_min - delta_r_sample + s*delta_r_sample;
            double exp_ik = exp(-beta_ik*(r_sample-rho_ik)*(r_sample-rho_ik));
            mosbest.evaluate(2*ai*ri*r_sample, gradients);
            for (int l = 0; l != L_plus_1; ++l) {
                integrand_l_at_r(l,s) =
                    r_sample*r_sample*
                    mosbest._in[l]*
                    exp_ik;
                integrand_derivative_l_at_r(l,s) =
                    2*ai*r_sample*r_sample*r_sample*
                    mosbest._din[l]*
                    exp_ik;
            }
        }
        // ... integrate (à la Simpson)
        std::vector<double> &ints = *integrals;
        std::vector<double> &ints_deriv = *integrals_derivative;
        for (int s = 0; s < n_steps; ++s) {
            for (int l = 0; l != L_plus_1; ++l) {
                ints[l] += delta_r_step/6.*(
                    integrand_l_at_r(l, 2*s)+
                    4*integrand_l_at_r(l, 2*s+1)+
                    integrand_l_at_r(l, 2*s+2)
                );
                ints_deriv[l] += delta_r_step/6.*(
                    integrand_derivative_l_at_r(l, 2*s)+
                    4*integrand_derivative_l_at_r(l, 2*s+1)+
                    integrand_derivative_l_at_r(l, 2*s+2)
                );
            }
        }
    }
    else {
        // Compute samples ...
        ub::matrix<double> integrand_l_at_r = ub::zero_matrix<double>(L_plus_1, n_sample);
        for (int s = 0; s < n_sample; ++s) {
            double r_sample = r_min - delta_r_sample + s*delta_r_sample;
            double exp_ik = exp(-beta_ik*(r_sample-rho_ik)*(r_sample-rho_ik));
            mosbest.evaluate(2*ai*ri*r_sample, gradients);
            for (int l = 0; l != L_plus_1; ++l) {
                integrand_l_at_r(l,s) =
                    r_sample*r_sample*
                    mosbest._in[l]*
                    exp_ik;
            }
        }
        // ... integrate (à la Simpson)
        std::vector<double> &ints = *integrals;
        for (int s = 0; s < n_steps; ++s) {
            for (int l = 0; l != L_plus_1; ++l) {
                ints[l] += delta_r_step/6.*(
                    integrand_l_at_r(l, 2*s)+
                    4*integrand_l_at_r(l, 2*s+1)+
                    integrand_l_at_r(l, 2*s+2)
                );
            }
        }
    }

    return;
}

void RadialBasisGaussian::computeCoefficients(
        vec d,
        double r,
        double particle_sigma,
        radcoeff_t &Gnl,
        radcoeff_t *dGnl_dx,
        radcoeff_t *dGnl_dy,
        radcoeff_t *dGnl_dz) {

    bool gradients = false;
    if (dGnl_dx) {
        assert(dGnl_dy != NULL && dGnl_dz != NULL);
        gradients = true;
    }

	// Delta-type expansion =>
	// Second (l) dimension of <save_here> and <particle_sigma> ignored here
	if (particle_sigma < RadialBasis::RADZERO) {
	    if (gradients) {
            throw soap::base::NotImplemented("<RadialBasisGaussian::computeCoefficients> Gradients when sigma=0.");
        }
		basis_it_t it;
		int n = 0;
		for (it = _basis.begin(), n = 0; it != _basis.end(); ++it, ++n) {
			double gn_at_r = (*it)->at(r);
			for (int l = 0; l != Gnl.size2(); ++l) {
			    Gnl(n, l) = gn_at_r;
			}
		}
		Gnl = ub::prod(_Tij, Gnl);
	}
	else {
		// Particle properties
		double ai = 1./(2*particle_sigma*particle_sigma);
		double ri = r;
		SphericalGaussian gi_sph(vec(0,0,0), particle_sigma); // <- position should not matter, as only normalization used here
		double norm_g_dV_sph_i = gi_sph._norm_g_dV;

		int k = 0;
		basis_it_t it;
		for (it = _basis.begin(), k = 0; it != _basis.end(); ++it, ++k) {
			// Prefactor (r-independent)
			double ak = (*it)->_alpha;
			double rk = (*it)->_r0;
			double norm_r2_g2_dr_rad_k = (*it)->_norm_r2_g2_dr;
			double beta_ik = ai+ak;
			double rho_ik = ak*rk/beta_ik;
			double prefac =
			    4*M_PI *
				norm_r2_g2_dr_rad_k*norm_g_dV_sph_i *
				exp(-ai*ri*ri) *
				exp(-ak*rk*rk*(1-ak/beta_ik));

			// Zero coeffs (to be safe ...)
			for (int l = 0; l != Gnl.size2(); ++l) {
			    Gnl(k, l) = 0.0;
			}
			if (gradients) {
			    for (int l = 0; l != Gnl.size2(); ++l) {
			        (*dGnl_dx)(k,l) = 0.0;
			        (*dGnl_dy)(k,l) = 0.0;
			        (*dGnl_dz)(k,l) = 0.0;
			    }
			}

			// Compute integrals S r^2 dr i_l(2*ai*ri*r) exp(-beta_ik*(r-rho_ik)^2)
			// and (derivative) S 2*ai*r^3 dr i_l(2*ai*ri*r) exp(-beta_ik*(r-rho_ik)^2)
			if (gradients) {
			    std::vector<double> integrals;
                integrals.resize(Gnl.size2(), 0.);
                std::vector<double> integrals_derivative;
			    integrals_derivative.resize(Gnl.size2(), 0.);
			    compute_integrals_il_expik_r2_dr(
                    ai, ri, beta_ik, rho_ik, Gnl.size2(), _integration_steps,
                    &integrals, &integrals_derivative);
                for (int l = 0; l != Gnl.size2(); ++l) {
                    Gnl(k, l) = prefac*integrals[l];
                    double dgkl = -2.*ai*ri*Gnl(k,l) + prefac*integrals_derivative[l];
                    (*dGnl_dx)(k,l) = dgkl*d.getX();
                    (*dGnl_dy)(k,l) = dgkl*d.getY();
                    (*dGnl_dz)(k,l) = dgkl*d.getZ();
                }
			}
			else {
			    std::vector<double> integrals;
                integrals.resize(Gnl.size2(), 0.);
                compute_integrals_il_expik_r2_dr(
                    ai, ri, beta_ik, rho_ik, Gnl.size2(), _integration_steps,
                    &integrals, NULL);
                for (int l = 0; l != Gnl.size2(); ++l) {
                    Gnl(k, l) = prefac*integrals[l];
                }
			}
		}

//		for (int k = 0; k < (*dGnl_dx).size1(); ++k) {
//		    for (int l = 0; l < (*dGnl_dx).size2(); ++l) {
//		        std::cout << boost::format("%1$+1.7f %2$d %3$d %4$+1.7e %5$+1.7e %6$+1.7e %7$+1.7e")
//		            % r % k % l % Gnl(k,l) % (*dGnl_dx)(k,l) % (*dGnl_dy)(k,l) % (*dGnl_dz)(k,l) << std::endl;
//		    }
//		}

		Gnl = ub::prod(_Tij, Gnl);
		if (gradients) {
		    (*dGnl_dx) = ub::prod(_Tij, *dGnl_dx);
		    (*dGnl_dy) = ub::prod(_Tij, *dGnl_dy);
		    (*dGnl_dz) = ub::prod(_Tij, *dGnl_dz);
		}
	}
    return;
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

