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

FTSpectrum::FTSpectrum(Structure &structure, Options &options)
    : _structure(&structure), _options(&options) {
    return;
}

FTSpectrum::~FTSpectrum() {
    return;
}

void FTSpectrum::compute() {

    int L = 3;

    GLOG() << "Legendre ..." << std::endl;
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
