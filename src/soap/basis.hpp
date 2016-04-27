#ifndef _SOAP_BASIS_HPP
#define _SOAP_BASIS_HPP

#include <string>
#include <math.h>
#include <vector>
#include <fstream>

#include "soap/base/exceptions.hpp"
#include "soap/options.hpp"
#include "soap/globals.hpp"
#include "soap/structure.hpp"
#include "soap/angularbasis.hpp"
#include "soap/radialbasis.hpp"
#include "soap/cutoff.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

class Basis
{
public:
	Basis(Options *options);
	Basis() : _options(NULL), _radbasis(NULL), _angbasis(NULL), _cutoff(NULL) {;}
	~Basis();

	RadialBasis *getRadBasis() { return _radbasis; }
	AngularBasis *getAngBasis() { return _angbasis; }
	CutoffFunction *getCutoff() { return _cutoff; }
	const int &N() { return _radbasis->N(); }
	const int &L() { return _angbasis->L(); }

	static void registerPython();

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & _options;
		arch & _radbasis;
		arch & _angbasis;
		arch & _cutoff;
		return;
	}
private:
	Options *_options;
	RadialBasis *_radbasis;
	AngularBasis *_angbasis;
	CutoffFunction *_cutoff;
};


class BasisExpansion
{
public:
	typedef ub::matrix< std::complex<double> > coeff_t;
	typedef ub::zero_matrix< std::complex<double> > coeff_zero_t;

	BasisExpansion(Basis *basis);
	BasisExpansion() : _basis(NULL), _radbasis(NULL), _angbasis(NULL), _has_coeff(false), _has_coeff_grad(false) {;}
    ~BasisExpansion();

    Basis *getBasis() { return _basis; }
    coeff_t &getCoefficients() { return _coeff; }
    RadialBasis::radcoeff_t &getRadCoeffs() { return _radcoeff; }
	AngularBasis::angcoeff_t &getAngCoeffs() { return _angcoeff; }

	void computeCoefficients(double r, vec d);
    void computeCoefficients(double r, vec d, double weight, double weight_scale, double sigma, bool gradients);
    bool hasCoefficients() { return _has_coeff; }
    bool hasCoefficientsGrad() { return _has_coeff_grad; }
    void add(BasisExpansion &other) { _coeff = _coeff + other._coeff; }
    void conjugate();
    void writeDensity(std::string filename, Options *options,
        	Structure *structure, Particle *center);
    void writeDensityOnGrid(std::string filename, Options *options,
    	Structure *structure, Particle *center, bool fromExpansion);

    void setCoefficientsNumpy(boost::python::object &array);
    boost::python::object getCoefficientsNumpy();
    static void registerPython();

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _basis;
    	arch & _radbasis;
    	arch & _angbasis;

    	arch & _has_coeff;
    	arch & _coeff;

    	arch & _has_coeff_grad;
    	arch & _coeff_grad_x;
    	arch & _coeff_grad_y;
    	arch & _coeff_grad_z;
    }

private:
    Basis *_basis;
	RadialBasis *_radbasis;
	AngularBasis *_angbasis;

	bool _has_coeff;
	RadialBasis::radcoeff_t _radcoeff; // access via (n,l)
	AngularBasis::angcoeff_t _angcoeff; // access via (l*l+l+m)
	coeff_t _coeff; // access via (n, l*l+l+m)

	bool _has_coeff_grad;
	vec _weight_scale_grad;
	RadialBasis::radcoeff_t _radcoeff_grad_x;
	RadialBasis::radcoeff_t _radcoeff_grad_y;
	RadialBasis::radcoeff_t _radcoeff_grad_z;
	AngularBasis::angcoeff_t _angcoeff_grad_x;
	AngularBasis::angcoeff_t _angcoeff_grad_y;
	AngularBasis::angcoeff_t _angcoeff_grad_z;
	coeff_t _coeff_grad_x;
	coeff_t _coeff_grad_y;
	coeff_t _coeff_grad_z;
};

}

#endif
