#ifndef _SOAP_ANGULARBASIS_HPP
#define _SOAP_ANGULARBASIS_HPP

#include <string>
#include <math.h>
#include <vector>

#include "soap/base/exceptions.hpp"
#include "soap/base/objectfactory.hpp"
#include "soap/options.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;


class AngularBasis
{
public:
	typedef ub::vector< std::complex<double> > angcoeff_t;
	typedef ub::zero_vector< std::complex<double> > angcoeff_zero_t;

	AngularBasis() : _type("spherical-harmonic"), _L(0) {;}
	virtual ~AngularBasis() {;}

	std::string &identify() { return _type; }
	const int &L() { return _L; }
    virtual void configure(Options &options);
    virtual void computeCoefficients(vec d, double r, angcoeff_t &save_here);

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _type;
    	arch & _L;
    }

protected:
    std::string _type;
    int _L;
    static constexpr double RADZERO = 1e-10;
};


class AngularBasisFactory : public soap::base::ObjectFactory<std::string, AngularBasis>
{
private:
    AngularBasisFactory() {}
public:
    static void registerAll(void);
    AngularBasis *create(const std::string &key);
    friend AngularBasisFactory &AngularBasisOutlet();
};

inline AngularBasisFactory &AngularBasisOutlet() {
    static AngularBasisFactory _instance;
    return _instance;
}

inline AngularBasis *AngularBasisFactory::create(const std::string &key) {
    assoc_map::const_iterator it(getObjects().find(key));
    if (it != getObjects().end()) {
        AngularBasis *basis = (it->second)();
        return basis;
    }
    else {
        throw std::runtime_error("Factory key " + key + " not found.");
    }
}

} /* CLOSE NAMESPACE */

#endif
