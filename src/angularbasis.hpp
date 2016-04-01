#ifndef _SOAP_ANGULARBASIS_HPP
#define _SOAP_ANGULARBASIS_HPP

#include <string>
#include <math.h>
#include <vector>

#include "base/exceptions.hpp"
#include "base/objectfactory.hpp"
#include "options.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

class AngularCoefficients : public ub::vector< std::complex<double> >
{
public:

	AngularCoefficients(int L) : _L(L) {
        this->resize((L+1)*(L+1));
        for (int i=0; i!=size(); ++i) {
        	(*this)[i] = std::complex<double>(0,0);
        }
    }
	void set(int l, int m, std::complex<double> c) {
		if (this->checkSize(l, m)) (*this)[l*l+l+m] = c;
		else throw soap::base::OutOfRange("AngularCoefficients::set");
	}
	std::complex<double> &get(int l, int m) {
		if (this->checkSize(l, m)) return (*this)[l*l+l+m];
		else throw soap::base::OutOfRange("AngularCoefficients::get");
	}
	bool checkSize(int l, int m) { return (std::abs(m) <= l && l <= _L); }
	void conjugate() {
		for (int i = 0; i != size(); ++i) {
			(*this)[i] = std::conj((*this)[i]);
		}
	}
protected:
	int _L;
};


class AngularBasis
{
public:
	typedef ub::vector< std::complex<double> > angcoeff_t;
	typedef ub::zero_vector< std::complex<double> > angcoeff_zero_t;

	std::string &identify() { return _type; }
	const int &L() { return _L; }
    AngularBasis() : _type("spherical-harmonic"), _L(0) {;}
    virtual ~AngularBasis() {;}
    virtual void configure(Options &options);
    virtual void computeCoefficients(vec d, double r, angcoeff_t &save_here);
    virtual AngularCoefficients computeCoefficients(vec d, double r);
    virtual AngularCoefficients computeCoefficientsAllZero();

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


class AngularBasisFactory
    : public soap::base::ObjectFactory<std::string, AngularBasis>
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

}

#endif
