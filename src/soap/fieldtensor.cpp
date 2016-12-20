#include "soap/fieldtensor.hpp"
#include "soap/functions.hpp"
#include <boost/math/special_functions/legendre.hpp>

namespace soap {

namespace ub = boost::numeric::ublas;

// ============================
// Spectrum::FTSpectrum::Atomic
// ============================

const std::string AtomicSpectrumFT::_numpy_t = "float64";

AtomicSpectrumFT::AtomicSpectrumFT(Particle *center, int S)
    : _center(center), _S(S) {
    this->_Q0 = coeff_zero_t(S,1);
    this->_Q1 = coeff_zero_t(S,S);
    this->_Q2 = coeff_zero_t(S,S*S);
    this->_Q3 = coeff_zero_t(S,S*S*S);

    this->_s = center->getTypeId()-1;
    assert(_s >= 0 && "Type-IDs should start from 1");
}

AtomicSpectrumFT::~AtomicSpectrumFT() {
    _Q0.clear();
    _Q1.clear();
    _Q2.clear();
    _Q3.clear();
}

void AtomicSpectrumFT::registerPython() {
    using namespace boost::python;
    class_<AtomicSpectrumFT, AtomicSpectrumFT*>("AtomicSpectrumFT", init<Particle*, int>())
        .def("getCenter", &AtomicSpectrumFT::getCenter, return_value_policy<reference_existing_object>());
}

// ===================
// FieldTensorSpectrum
// ===================

std::vector<double> calculate_Alm(int L) {
    // A_{lm} = \sqrt{(l+m)!*(l-m)!}
    std::vector<double> out;
    out.resize((L+1)*(L+1), 0.0);
    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= L; ++m) {
            out[l*l+l+m] = sqrt(factorial(l+m)*factorial(l-m));
        }
    }
    return out;
}

std::vector<double> calculate_Blm(int L) {
    // B_{lm} = A_{lm} / (2l-1)!!
    std::vector<double> out;
    out.resize((L+1)*(L+1), 0.0);
    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= L; ++m) {
            out[l*l+l+m] = sqrt(factorial(l+m)*factorial(l-m))/factorial2(2*l-1);
        }
    }
    return out;
}

void calculate_Fl(double r, double a, int L, std::vector<double> &fl) {
    // See Elking et al.: "Gaussian Multipole Model", JCTC (2010), Eq. 15a++
    fl.clear();
    fl.resize(L+1, 0.0);
    if (r < 1e-10) { // TODO Define space quantum
        double a2 = a*a;
        double a_l = 1./a;
        int two_l = 1;
        int sign_l = -1;
        double rsqrt_pi = 1./sqrt(M_PI);
        for (int l = 0; l <= L; ++l) {
            a_l *= a2;
            two_l *= 2;
            sign_l *= -1;
            fl[l] = sign_l*two_l*a_l*rsqrt_pi/(2*l+1);
        }
    }
    else {
        // Initialise
        fl[0] = std::erf(a*r)/r;
        double r2 = r*r;
        double a2 = a*a;
        double r_sqrt_pi_exp_a2r2 = 1./sqrt(M_PI) * exp(-a2*r2);
        double al = 1./a;
        double tl = 1.;
        // Compute
        for (int l = 1; l <= L; ++l) {
            al *= a2;
            tl *= 2;
            fl[l] = -1./r2 * ( (2*l-1)*fl[l-1] + pow(-1,l)*tl*al*r_sqrt_pi_exp_a2r2 );
        }
    }

    return;
}

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

    void computeTlmlm(vec d12, double r12, double a12, int L1, int L2, coeff_t &T) {
        T.clear();
        T = coeff_zero_t((L1+1)*(L1+1), (L2+1)*(L2+1));

        int Lg = (L1 > L2) ? L1 : L2;
        int L12 = L1+L2;
        // Compute Rlm's
        std::vector<std::complex<double>> Rlm;
        calculate_solidharm_Rlm(d12, r12, Lg, Rlm);
        // Compute Fl's
        std::vector<double> Fl;
        calculate_Fl(r12, a12, L12, Fl);

        for (int l1 = 0; l1 <= L1; ++l1) {
        for (int m1 = -l1; m1 <= l1; ++m1) {
            int lm1 = l1*l1+l1+m1;
        for (int l2 = 0; l2 <= L2; ++l2) {
        for (int m2 = -l2; m2 <= l2; ++m2) {
            int lm2 = l2*l2+l2+m2;

            std::complex<double> tlm1lm2 = 0.0;
            int lg = (l1 > l2) ? l1 : l2;

            for (int l = 0; l <= lg; ++l) {
            for (int m = -l; m <= l; ++m) {

                if (std::abs(m2+m) > l2-l) continue;
                if (std::abs(m1-m) > l1-l) continue;
                int lm = l*l+l+m;
                tlm1lm2 +=
                    pow(-1, l1+m)*factorial2(2*l-1)/(Alm[lm]*Alm[lm])
                  * std::conj( Rlm[ (l2-l)*(l2-l) + (l2-l) + m2+m ] )
                  * std::conj( Rlm[ (l1-l)*(l1-l) + (l1-l) + m1-m ] )
                  * Fl[ l1+l2-l ];

            }} // l m

            tlm1lm2 *=
                Alm[lm1]*Alm[lm1] * Alm[lm2]*Alm[lm2]
              / factorial2(2*l1-1)*factorial2(2*l2-1);

             // TODO Assert .imag() == zero // TODO Does not actually need to be zero!?
             T(lm1,lm2) = tlm1lm2;
        }}}} // l1 m1 l2 m2
        return;
    }

    double computeTl1m1l2m2(double d12, double r12, double a12, int l1, int l2) {

    }
private:
    std::vector<double> Alm;
    std::vector<double> Blm;
};


void calculate_r_dr_erfar_r(double r, double a, int L, bool normalise, std::vector<double> &cl) {
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
}


FTSpectrum::FTSpectrum(Structure &structure, Options &options)
    : _structure(&structure), _options(&options) {
    _cutoff = CutoffFunctionOutlet().create(_options->get<std::string>("radialcutoff.type"));
	_cutoff->configure(*_options);
    return;
}

FTSpectrum::~FTSpectrum() {
    return;
}

void FTSpectrum::compute() {
    GLOG() << "Computing FTSpectrum ..." << std::endl;

    int L = 3; // TODO
    int S = 4;

    // CREATE ATOMIC SPECTRA
    for (auto it = _atomic_array.begin(); it != _atomic_array.end(); ++it) {
        delete *it;
    }
    _atomic_array.clear();
    for (auto pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
        atomic_t *new_atomic = new atomic_t(*pit, S);
        _atomic_array.push_back(new_atomic);
    }

    for (auto it1 = beginAtomic(); it1 != endAtomic(); ++it1) {
        atomic_t *a = *it1;
        int sa = a->getTypeIdx();
        vec ra = a->getCenter()->getPos();
        for (auto it2 = beginAtomic(); it2 != endAtomic(); ++it2) {
            atomic_t *b = *it2;
            int sb = b->getTypeIdx();
            vec rb = b->getCenter()->getPos();
            // Apply weight function
            vec dr_ab = _structure->connect(ra, rb);
            double R_ab = soap::linalg::abs(dr_ab);
            vec d_ab = dr_ab/R_ab;
            if (! _cutoff->isWithinCutoff(R_ab)) continue;
            double w_ab = _cutoff->calculateWeight(R_ab);
            GLOG() << "    " << a->getCenter()->getId() << ":" << b->getCenter()->getId() << " R=" << R_ab << " w=" << w_ab << std::endl;
            // Interact
            // ...
        }
    }

    GLOG() << "Factorial2 ..." << std::endl;
    for (int n = 0; n < 16; ++n) {
        GLOG() << n << "!! = " << factorial2(n) << std::endl;
    }

    GLOG() << "Radial damping functions ..." << std::endl;
    std::vector<double> cl;
    calculate_Fl(0., 0.5, 5, cl);
    for (int l = 0; l <= 4; ++l) {
        GLOG() << l << " fl = " << cl[l] << std::endl;
    }
    calculate_Fl(0.5, 0.5, 5, cl);
    for (int l = 0; l <= 4; ++l) {
        GLOG() << l << " fl = " << cl[l] << std::endl;
    }
    calculate_Fl(4.5, 0.5, 5, cl);
    for (int l = 0; l <= 4; ++l) {
        GLOG() << l << " fl = " << cl[l] << std::endl;
    }
    calculate_Fl(10.5, 0.5, 5, cl);
    for (int l = 0; l <= 4; ++l) {
        GLOG() << l << " fl = " << cl[l] << std::endl;
    }

    GLOG() << "Solid harmonics Rlm ..." << std::endl;
    std::vector<std::complex<double>> rlm;
    double phi = 0.6;
    double theta = 0.7;
    double sp = std::sin(phi);
    double st = std::sin(theta);
    double cp = std::cos(phi);
    double ct = std::cos(theta);
    vec d = vec(st*cp, st*sp, ct);
    double r = 0.5;
    calculate_solidharm_Rlm(d, r, L, rlm);
    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {
            int lm = l*l+l+m;
            GLOG() << l << " " << m << " " << rlm[lm] << std::endl;
        }
    }
    GLOG() << "r = 0 ..." << std::endl;
    r = 0.;
    calculate_solidharm_Rlm(d, r, L, rlm);
    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {
            int lm = l*l+l+m;
            GLOG() << l << " " << m << " " << rlm[lm] << std::endl;
        }
    }

    /*GLOG() << "Legendre ..." << std::endl;
    std::vector<double> plm;
    calculate_legendre_plm(L, 0.3, plm);
    for (int l = 0; l <= L; ++l) {
        for (int m = 0; m <= l; ++m) {
            int lm = l*(l+1)/2 + m;
            GLOG() << l << " " << m << " " << plm[lm] << std::endl;
        }
    }
    GLOG() << "Factorial ..." << std::endl;
    for (int n = 0; n < 16; ++n) {
        GLOG() << n << "! = " << factorial(n) << std::endl;
    }

    GLOG() << "Solid harmonics ..." << std::endl;
    std::vector<std::complex<double>> rlm;
    std::vector<std::complex<double>> ilm;
    double phi = 0.6;
    double theta = 0.7;
    double sp = std::sin(phi);
    double st = std::sin(theta);
    double cp = std::cos(phi);
    double ct = std::cos(theta);
    vec d = vec(st*cp, st*sp, ct);
    double r = 0.5;
    calculate_solidharm_rlm_ilm(d, r, L, rlm, ilm);
    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {
            int lm = l*l+l+m;
            GLOG() << l << " " << m << " " << rlm[lm] << "  " << ilm[lm] << std::endl;
        }
    }


    */

    return;
}

void FTSpectrum::registerPython() {
    using namespace boost::python;
    class_<FTSpectrum>("FTSpectrum", init<Structure &, Options &>())
        .def("__iter__", range<return_value_policy<reference_existing_object> >(&FTSpectrum::beginAtomic, &FTSpectrum::endAtomic))
        .def("compute", &FTSpectrum::compute);
    return;
}


}
