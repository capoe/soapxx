#ifndef _SOAP_FUNCTIONS_H
#define	_SOAP_FUNCTIONS_H

#include <math.h>
#include <vector>

#include "soap/types.hpp"

namespace soap {

struct RadialGaussian
{
	RadialGaussian(double r0, double sigma);
	RadialGaussian();
	double at(double r);

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & _r0;
		arch & _sigma;
		arch & _alpha;
		arch & _norm_r2_g2_dr;
		arch & _norm_r2_g_dr;
	}

	double _r0;
	double _sigma;
	double _alpha;
	// Integral S g^2 r^2 dr
	double _integral_r2_g2_dr;
	double _norm_r2_g2_dr;
	// Integral S g r^2 dr
	double _integral_r2_g_dr;
	double _norm_r2_g_dr;
};

struct SphericalGaussian
{
	SphericalGaussian(vec r0, double sigma);

	vec _r0;
	double _sigma;
	double _alpha;
	double _norm_g_dV;
};

struct ModifiedSphericalBessel1stKind
{
    ModifiedSphericalBessel1stKind(int degree);
    void evaluate(double r, bool differentiate);

	static std::vector<double> eval(int degree, double r);
	static constexpr double RADZERO = 1e-10;
	static constexpr double SPHZERO = 1e-4;

    int _degree;
    std::vector<double> _in;
    std::vector<double> _din;
};

struct GradSphericalYlm
{
    typedef std::complex<double> cmplx;
    static std::vector<cmplx > eval(int l, int m, vec &r);

    static constexpr double RADZERO = 1e-10;
};

std::complex<double> pow_nnan(std::complex<double> z, double a);
int factorial(int n);

}

#endif
