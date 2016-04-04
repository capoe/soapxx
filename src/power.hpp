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

    PowerExpansion() {
        ;
    }
	PowerExpansion(BasisExpansion *basex) {
        ;
    }

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
        ;
    }

private:

	coeff_t _coeff; // access via (N*n+k, l) with shape (N*N, L+1)
};

}

#endif
