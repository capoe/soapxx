#ifndef _SOAP_COULOMB_HPP
#define _SOAP_COULOMB_HPP

#include "soap/structure.hpp"
#include "soap/options.hpp"
#include "soap/cutoff.hpp"

namespace soap {

class AtomicSpectrumHC
{
public:
    typedef double dtype_t;
    typedef ub::matrix<dtype_t> coeff_t;
    typedef ub::zero_matrix<dtype_t> coeff_zero_t;
    static const std::string _numpy_t;

    AtomicSpectrumHC(Particle *center, int S);
   ~AtomicSpectrumHC();
    Particle *getCenter() { return _center; }
    int getTypeIdx() { return _s; }

    static void registerPython();
    void setCoefficientsNumpy_k1(boost::python::object &np_array);
    void setCoefficientsNumpy_k2(boost::python::object &np_array);
    void setCoefficientsNumpy_k3(boost::python::object &np_array);
    void setCoefficientsNumpy_k4(boost::python::object &np_array);
    boost::python::object getCoefficientsNumpy_k1();
    boost::python::object getCoefficientsNumpy_k2();
    boost::python::object getCoefficientsNumpy_k3();
    boost::python::object getCoefficientsNumpy_k4();

    coeff_t _Q0;
    coeff_t _Q1;
    coeff_t _Q2;
    coeff_t _Q3;

private:
    Particle *_center;
    int _S; // <- Number of distinct types
    int _s; // <- Type index
};

class HierarchicalCoulomb
{
public:
    typedef AtomicSpectrumHC atomic_t;
    typedef std::vector<AtomicSpectrumHC*> atomic_array_t;
	typedef std::vector<AtomicSpectrumHC*>::iterator atomic_it_t;

    HierarchicalCoulomb(Structure &structure, Options &options);
   ~HierarchicalCoulomb();
    void compute();
    atomic_it_t beginAtomic() { return _atomic_array.begin(); }
    atomic_it_t endAtomic() { return _atomic_array.end(); }

    static void registerPython();

private:
    Structure *_structure;
    Options *_options;
    CutoffFunction *_cutoff;
    int _S;

    atomic_array_t _atomic_array;
};

}






#endif /* _SOAP_COULOMB_HPP */
