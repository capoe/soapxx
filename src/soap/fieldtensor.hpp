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

    static const std::string _numpy_t;

    // FIELD MOMENTS
    typedef std::complex<double> cmplx_t;
    typedef ub::matrix<cmplx_t> field_t; // (l'l'', lm)
    typedef ub::zero_matrix<cmplx_t> field_zero_t;
    typedef std::map<std::string, field_t*> field_map_t; // (type)->(l'l''..., lm)
    typedef std::vector<field_map_t> body_map_t; // (k)->(type)->(l'l''..., lm)

    // CONTRACTIONS
    typedef std::pair<std::string, std::string> channel_t;
    typedef ub::matrix<cmplx_t> coeff_t; // (k=1:0 k=1:l' k=2:l'l'', l)
    typedef ub::zero_matrix<cmplx_t> coeff_zero_t;
    typedef std::map<channel_t, coeff_t*> coeff_map_t;

    AtomicSpectrumFT(Particle *center, int K, int L);
   ~AtomicSpectrumFT();
    Particle *getCenter() { return _center; }
    std::string getType() { return _type; }
    int getTypeIdx() { return _s; }

    static void registerPython();

    body_map_t _body_map;
    coeff_map_t _coeff_map;

    void addField(int k, std::string type, field_t &flm);
    field_t *getField(int k, std::string type);
    field_t *getCreateField(int k, std::string type, int s1, int s2);
    field_map_t &getFieldMap(int k) { return _body_map[k]; }
    coeff_t *getCreateContraction(channel_t &channel, int size1, int size2);
    coeff_map_t &getCoeffMap() { return _coeff_map; }
    void contract();

private:
    Particle *_center;
    int _K; // <- Body-order cutoff
    int _L; // <- Angular momentum cutoff
    int _s; // <- Type index
    std::string _type; // <- Type string
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
