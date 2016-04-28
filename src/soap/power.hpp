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

	PowerExpansion() : _basis(NULL), _L(-1), _N(-1), _has_scalars(false), _has_gradients(false) {;}
    PowerExpansion(Basis *basis);

    Basis *getBasis() { return _basis; }
    coeff_t &getCoefficients() { return _coeff; }
    void computeCoefficients(BasisExpansion *basex1, BasisExpansion *basex2);
    void computeCoefficientsGradients(BasisExpansion *dqnlm, BasisExpansion *qnlm, bool same_types);
    void add(PowerExpansion *other);
    void writeDensity(std::string filename, Options *options, Structure *structure, Particle *center);

    void setCoefficientsNumpy(boost::python::object &np_array);
    boost::python::object getCoefficientsNumpy();
    static void registerPython();

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

    bool _has_scalars;
	coeff_t _coeff; // access via (N*n+k, l) with shape (N*N, L+1)

	bool _has_gradients;
	coeff_t _coeff_grad_x;
	coeff_t _coeff_grad_y;
	coeff_t _coeff_grad_z;
};

}

#endif
