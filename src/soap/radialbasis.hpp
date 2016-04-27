#ifndef _SOAP_RADIALBASIS_HPP
#define _SOAP_RADIALBASIS_HPP

#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include "soap/base/exceptions.hpp"
#include "soap/base/objectfactory.hpp"
#include "soap/options.hpp"
#include "soap/functions.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

// CLASSES IN THIS HEADER
// RadialBasis
// RadialBasisGaussian
// RadialBasisLegendre
// RadialBasisFactory

class RadialBasis
{
public:
	typedef ub::matrix<double> radcoeff_t;
	typedef ub::zero_matrix<double> radcoeff_zero_t;

	virtual ~RadialBasis() {;}

	std::string &identify() { return _type; }
	const int &N() { return _N; }

    virtual void configure(Options &options);
    virtual void computeCoefficients(
        vec d,
        double r,
        double particle_sigma,
        radcoeff_t &Gnl,
        radcoeff_t *dGnl_dx,
        radcoeff_t *dGnl_dy,
        radcoeff_t *dGnl_dz);

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _type;
    	arch & _N;
    	arch & _Rc;
    	arch & _integration_steps;
    	arch & _mode;
    }

protected:
   std::string _type;
   int _N;
   double _Rc;
   int _integration_steps;
   std::string _mode; // <- 'equispaced' or 'adaptive'
   bool _is_ortho;

   static constexpr double RADZERO = 1e-10;
};


class RadialBasisGaussian : public RadialBasis
{
public:
	typedef RadialGaussian basis_fct_t;
	typedef std::vector<RadialGaussian*> basis_t;
	typedef basis_t::iterator basis_it_t;

    RadialBasisGaussian();
   ~RadialBasisGaussian() { this->clear();
    }
    void clear();
    void configure(Options &options);
    void computeCoefficients(
        vec d,
        double r,
        double particle_sigma,
        radcoeff_t &Gnl,
        radcoeff_t *dGnl_dx,
        radcoeff_t *dGnl_dy,
        radcoeff_t *dGnl_dz);

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & boost::serialization::base_object<RadialBasis>(*this);
    	arch & _sigma;
    	arch & _basis;
    	arch & _Sij;
    	arch & _Uij;
    	arch & _Tij;
    }

protected:
    double _sigma;
    basis_t _basis;

    // ORTHONORMALIZATION
    ub::matrix<double> _Sij; // Overlap
    ub::matrix<double> _Uij; // Cholesky factor
    ub::matrix<double> _Tij; // Transformation matrix
};


class RadialBasisLegendre : public RadialBasis
{
public:
	RadialBasisLegendre() {
		_type = "legendre";
		_is_ortho = true;
	}
};

void compute_integrals_il_expik_r2_dr(
    double ai,
    double ri,
    double beta_ik,
    double rho_ik,
    int L_plus_1,
    int n_steps,
    std::vector<double> *integrals,
    std::vector<double> *integrals_derivative);

class RadialBasisFactory 
    : public soap::base::ObjectFactory<std::string, RadialBasis>
{
private:
    RadialBasisFactory() {}
public:
    static void registerAll(void);
    RadialBasis *create(const std::string &key);
    friend RadialBasisFactory &RadialBasisOutlet();
};

inline RadialBasisFactory &RadialBasisOutlet() {
    static RadialBasisFactory _instance;
    return _instance;
}

inline RadialBasis *RadialBasisFactory::create(const std::string &key) {
    assoc_map::const_iterator it(getObjects().find(key));
    if (it != getObjects().end()) {
        RadialBasis *basis = (it->second)();
        return basis;
    } 
    else {
        throw std::runtime_error("Factory key " + key + " not found.");
    }
}

} /* CLOSE NAMESPACE */

BOOST_CLASS_EXPORT_KEY(soap::RadialBasis);
BOOST_CLASS_EXPORT_KEY(soap::RadialBasisGaussian);

#endif
