#ifndef _SOAP_RADIALBASIS_HPP
#define _SOAP_RADIALBASIS_HPP

#include "base/objectfactory.hpp"
#include "options.hpp"
#include <string>
#include <boost/numeric/ublas/matrix.hpp>

namespace soap {

namespace ub = boost::numeric::ublas;

class RadialBasis
{
public:
	std::string &identify() { return _type; }
    virtual ~RadialBasis() {;}
    virtual void configure(Options &options);

protected:
   bool _is_ortho;
   std::string _type;
   int _N;
   double _Rc;
};

struct RadialGaussian
{
	RadialGaussian(double r0, double sigma);
	double at(double r);
	double _r0;
	double _sigma;
	double _alpha;
	double _integral_4_pi_r2_g2_dr;
	double _integral_r2_g2_dr;
	double _norm_dV;
};

class RadialBasisGaussian : public RadialBasis
{
public:
	typedef RadialGaussian basis_fct_t;
	typedef std::vector<RadialGaussian*> basis_t;
	typedef basis_t::iterator basis_it_t;

    RadialBasisGaussian() {
        _type = "gaussian";
        _is_ortho = false;
        _sigma = 0.5;
    }
   ~RadialBasisGaussian() {
        this->clear();
    }
    void clear() {
    	for (basis_it_t bit=_basis.begin(); bit!=_basis.end(); ++bit) {
			delete *bit;
		}
    	_basis.clear();
    }
    void configure(Options &options);
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







}

#endif
