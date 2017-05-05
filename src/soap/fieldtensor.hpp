#ifndef _SOAP_FIELDTENSOR_HPP
#define _SOAP_FIELDTENSOR_HPP

#include "soap/structure.hpp"
#include "soap/options.hpp"
#include "soap/cutoff.hpp"
#include "soap/functions.hpp"
#include "boost/multi_array.hpp"

namespace soap {

class AtomicSpectrumFT
{
public:
    typedef double dtype_t;
    static const std::string _numpy_t;

    // FIELD MOMENTS
    typedef std::complex<double> cmplx_t;
    typedef ub::matrix<cmplx_t> field_t; // (l'l'', lm)
    typedef ub::zero_matrix<cmplx_t> field_zero_t;
    typedef std::map<std::string, field_t*> field_map_t; // (type)->(l'l''..., lm)
    typedef std::vector<field_map_t> body_map_t; // (k)->(type)->(l'l''..., lm)

    // CONTRACTIONS
    typedef std::pair<std::string, std::string> channel_t;
    typedef ub::matrix<dtype_t> coeff_t; // (k=1:0 k=1:l' k=2:l'l'', l)
    typedef ub::zero_matrix<dtype_t> coeff_zero_t;
    typedef std::map<channel_t, coeff_t*> coeff_map_t;

    // PHYSICAL
    field_t _F;
    field_t _M;
    std::vector<double> _alpha;

    AtomicSpectrumFT(Particle *center, int K, int L);
   ~AtomicSpectrumFT();
    Particle *getCenter() { return _center; }
    std::string getType() { return _type; }
    boost::python::list getTypes();
    boost::python::object getCoefficientsNumpy(std::string s1, std::string s2);
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
    std::list<std::string> _nb_types;
};

class FTBasis
{
public:
    FTBasis(Options &options) : _K(-1), _L(-1) {};
   ~FTBasis() {};
private:
    int _K;
    int _L;
};

std::vector<double> calculate_Alm(int L);
std::vector<double> calculate_Blm(int L);
void calculate_Fl(double r, double a, int L, std::vector<double> &fl);

class Tlmlm
{
public:
    typedef std::complex<double> dtype_t;
    typedef ub::matrix< dtype_t > coeff_t;
    typedef ub::zero_matrix< dtype_t > coeff_zero_t;

    Tlmlm(int L) {
        Alm = calculate_Alm(L);
        Blm = calculate_Blm(L);
    }

    void computeTlmlm(vec d12, double r12, double a12, int L1, int L2, coeff_t &T);

private:
    std::vector<double> Alm;
    std::vector<double> Blm;
};

/*void calculate_r_dr_erfar_r(double r, double a, int L, bool normalise, std::vector<double> &cl) {
    // Derivatives (1/r d/dr)^l (erf(ar)/r) for 0 <= l <= L
    cl.clear();
    cl.resize(L+1, 0.0);
    // Initialise
    cl[0] = std::erf(a*r)/r;
    double r2 = r*r;
    double a2 = a*a;
    double r_sqrt_pi_exp_a2r2 = 1./sqrt(M_PI) * exp(-a2*r2);
    double al = 1./a;
    double tl = 1.;
    // Compute
    for (int l = 1; l <= L; ++l) {
        al *= a2;
        tl *= 2;
        cl[l] = 1./r2 * ( (2*l-1)*cl[l-1] - tl*al*r_sqrt_pi_exp_a2r2 );
    }

    if (normalise) {
        double rl = r;
        cl[0] *= rl; // * factorial2(-1), which equals 1
        for (int l = 1; l <= L; ++l) {
            rl *= r2;
            cl[l] *= rl / factorial2(2*l-1);
        }
    }

    return;
}*/

class FTSpectrum
{
public:
    typedef AtomicSpectrumFT atomic_t;
    typedef std::vector<AtomicSpectrumFT*> atomic_array_t;
	typedef std::vector<AtomicSpectrumFT*>::iterator atomic_it_t;

    FTSpectrum(Structure &structure, Options &options);
   ~FTSpectrum();
    void compute();
    void computeFieldTensors(std::map<int, std::map<int, Tlmlm::coeff_t> > &i1_i2_T12);
    void createAtomic();
    atomic_it_t beginAtomic() { return _atomic_array.begin(); }
    atomic_it_t endAtomic() { return _atomic_array.end(); }
    void energySCF(int k, std::map<int, std::map<int, Tlmlm::coeff_t> > &i1_i2_T12);
    void polarize();

    static void registerPython();

private:
    Structure *_structure;
    Options *_options;
    CutoffFunction *_cutoff;

    int _K;
    int _L;

    atomic_array_t _atomic_array;
};



}

#endif
