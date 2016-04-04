#ifndef _SOAP_BASIS_HPP
#define _SOAP_BASIS_HPP

#include <string>
#include <math.h>
#include <vector>
#include <fstream>

#include "base/exceptions.hpp"
#include "options.hpp"
#include "globals.hpp"
#include "structure.hpp"
#include "angularbasis.hpp"
#include "radialbasis.hpp"
#include "cutoff.hpp"

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
	BasisExpansion() : _basis(NULL), _radbasis(NULL), _angbasis(NULL) {;}
    ~BasisExpansion();

    Basis *getBasis() { return _basis; }
    coeff_t &getCoefficients() { return _coeff; }
    RadialBasis::radcoeff_t &getRadCoeffs() { return _radcoeff; }
	AngularBasis::angcoeff_t &getAngCoeffs() { return _angcoeff; }

    void computeCoefficients(double r, vec d, double weight, double sigma);
    void add(BasisExpansion &other) { _coeff = _coeff + other._coeff; }
    void conjugate();
    void writeDensityOnGrid(std::string filename, Options *options,
    	Structure *structure, Particle *center, bool fromExpansion);

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _basis;
    	arch & _radbasis;
    	arch & _angbasis;
    	arch & _coeff;
    }

private:
    Basis *_basis;
	RadialBasis *_radbasis;
	AngularBasis *_angbasis;

	RadialBasis::radcoeff_t _radcoeff; // access via (n,l)
	AngularBasis::angcoeff_t _angcoeff; // access via (l*l+l+m)
	coeff_t _coeff; // access via (n, l*l+l+m)
};

}

#endif
