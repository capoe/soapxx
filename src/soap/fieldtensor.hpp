#ifndef _SOAP_FIELDTENSOR_HPP
#define _SOAP_FIELDTENSOR_HPP

#include "soap/structure.hpp"
#include "soap/options.hpp"
#include "soap/cutoff.hpp"
#include "boost/multi_array.hpp"

namespace soap {

class AtomicSpectrumFT
{
public:
    typedef double dtype_t;
    typedef ub::matrix<dtype_t> coeff_t;
    typedef ub::zero_matrix<dtype_t> coeff_zero_t;
    static const std::string _numpy_t;

    AtomicSpectrumFT(Particle *center, int L, int S);
   ~AtomicSpectrumFT();
    Particle *getCenter() { return _center; }
    int getTypeIdx() { return _s; }

    static void registerPython();

    typedef ub::matrix<std::complex<double>> field_coeff_t;
    typedef ub::zero_matrix<std::complex<double>> field_coeff_zero_t;
    typedef field_coeff_t field0_t;
    typedef boost::multi_array<field0_t, 1> field1_t;
    typedef boost::multi_array<field0_t, 2> field2_t;
    typedef boost::multi_array<field0_t, 3> field3_t;
    //typedef std::vector< field0_t* > fieldn_t;

    typedef field0_t moment0_t;
    typedef field1_t moment1_t;
    typedef field2_t moment2_t;
    typedef field3_t moment3_t;

    field0_t _f0; // alm
    field1_t _f1; // l',alm
    field2_t _f2; // l''l',alm
    field3_t _f3; // l'''l''l',alm

    moment0_t _q0;
    moment1_t _q1;
    moment2_t _q2;
    moment3_t _q3;

    coeff_t _p0; // l,aa'
    coeff_t _p1; // ll',aa'
    coeff_t _p2; // ll'l'',aa'
    coeff_t _p3; // ll'l''l''',aa'

private:
    Particle *_center;
    int _L; // <- Angular momentum cutoff
    int _S; // <- Number of distinct types
    int _s; // <- Type index
};

class FTSpectrum
{
public:
    typedef AtomicSpectrumFT atomic_t;
    typedef std::vector<AtomicSpectrumFT*> atomic_array_t;
	typedef std::vector<AtomicSpectrumFT*>::iterator atomic_it_t;

    FTSpectrum(Structure &structure, Options &options);
   ~FTSpectrum();
    void compute();
    atomic_it_t beginAtomic() { return _atomic_array.begin(); }
    atomic_it_t endAtomic() { return _atomic_array.end(); }

    static void registerPython();

private:
    Structure *_structure;
    Options *_options;
    CutoffFunction *_cutoff;

    atomic_array_t _atomic_array;
};



}

#endif
