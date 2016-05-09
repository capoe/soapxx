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
	Options *getOptions() { return _options; }
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
	BasisExpansion() : _basis(NULL), _radbasis(NULL), _angbasis(NULL), _has_scalars(false), _has_gradients(false) {;}
    ~BasisExpansion();

    Basis *getBasis() { return _basis; }
    coeff_t &getCoefficients() { return _coeff; }
    coeff_t &getCoefficientsGradX() { return _coeff_grad_x; }
    coeff_t &getCoefficientsGradY() { return _coeff_grad_y; }
    coeff_t &getCoefficientsGradZ() { return _coeff_grad_z; }
    RadialBasis::radcoeff_t &getRadCoeffs() { return _radcoeff; }
	AngularBasis::angcoeff_t &getAngCoeffs() { return _angcoeff; }

	void computeCoefficients(double r, vec d);
    void computeCoefficients(double r, vec d, double weight, double weight_scale, double sigma, bool gradients);
    bool hasScalars() { return _has_scalars; }
    bool hasGradients() { return _has_gradients; }
    void add(BasisExpansion &other) { _coeff = _coeff + other._coeff; }
    void add(BasisExpansion &other, double scale) { _coeff = _coeff + scale*other._coeff; }
    void addGradient(BasisExpansion &other);
    void zeroGradient();
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

    	arch & _has_scalars;
    	arch & _coeff;

    	arch & _has_gradients;
    	arch & _coeff_grad_x;
    	arch & _coeff_grad_y;
    	arch & _coeff_grad_z;
    }

private:
    Basis *_basis;
	RadialBasis *_radbasis;
	AngularBasis *_angbasis;

	bool _has_scalars;
	RadialBasis::radcoeff_t _radcoeff; // access via (n,l)
	AngularBasis::angcoeff_t _angcoeff; // access via (l*l+l+m)
	coeff_t _coeff; // access via (n, l*l+l+m)

	bool _has_gradients;
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
