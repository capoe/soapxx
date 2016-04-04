#ifndef _SOAP_POWER_HPP
#define _SOAP_POWER_HPP

#include <string>
#include <math.h>
#include <vector>
#include <fstream>

#include "base/exceptions.hpp"
#include "basis.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;


class PowerExpansion
{
public:
	typedef ub::matrix< std::complex<double> > coeff_t;
	typedef ub::zero_matrix< std::complex<double> > coeff_zero_t;

	PowerExpansion() : _basis(NULL), _L(-1), _N(-1) {;}
    PowerExpansion(Basis *basis);

    void computeCoefficients(BasisExpansion *basex1, BasisExpansion *basex2);
    void add(PowerExpansion *other);

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        arch & _basis;
        arch & _N;
        arch & _L;
        arch & _coeff;
    }

private:
    Basis *_basis;
    int _N;
    int _L;
	coeff_t _coeff; // access via (N*n+k, l) with shape (N*N, L+1)
};

}

#endif
