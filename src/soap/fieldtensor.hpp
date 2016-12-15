#ifndef _SOAP_FIELDTENSOR_HPP
#define _SOAP_FIELDTENSOR_HPP

#include "soap/structure.hpp"
#include "soap/options.hpp"
#include "soap/cutoff.hpp"

namespace soap {

class AtomicSpectrumFT
{
public:
    typedef double dtype_t;
    typedef ub::matrix<dtype_t> coeff_t;
    typedef ub::zero_matrix<dtype_t> coeff_zero_t;
    static const std::string _numpy_t;

    AtomicSpectrumFT(Particle *center, int S);
   ~AtomicSpectrumFT();
    Particle *getCenter() { return _center; }
    int getTypeIdx() { return _s; }

    static void registerPython();

    coeff_t _Q0;
    coeff_t _Q1;
    coeff_t _Q2;
    coeff_t _Q3;

private:
    Particle *_center;
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
