#include "power.hpp"
#include "linalg/numpy.hpp"

namespace soap {

PowerExpansion::PowerExpansion(Basis *basis) :
    _basis(basis),
    _L(basis->getAngBasis()->L()),
    _N(basis->getRadBasis()->N()) {
    _coeff = coeff_zero_t(_N*_N, _L+1);
}

void PowerExpansion::computeCoefficients(BasisExpansion *basex1, BasisExpansion *basex2) {
    if (!_basis) throw soap::base::APIError("PowerExpansion::computeCoefficients, basis not initialised.");
    BasisExpansion::coeff_t &coeff1 = basex1->getCoefficients();
    BasisExpansion::coeff_t &coeff2 = basex2->getCoefficients();
    for (int n = 0; n < _N; ++n) {
        for (int k = 0; k < _N; ++k) {
            for (int l = 0; l < (_L+1); ++l) {
                //std::cout << n << " " << k << " " << l << " : " << std::flush;
                std::complex<double> c_nkl = 0.0;
                for (int m = -l; m <= l; ++m) {
                    //std::cout << m << " " << std::flush;
                    c_nkl += coeff1(n, l*l+l+m)*std::conj(coeff2(k, l*l+l+m));
                }
                _coeff(n*_N+k, l) = 2.*sqrt(2.)*M_PI/sqrt(2.*l+1)*c_nkl; // Normalization = sqrt(8\pi^2/(2l+1))
                //std::cout << std::endl;
            }
        }
    }
    //throw soap::base::APIError("");
    return;
}

void PowerExpansion::add(PowerExpansion *other) {
    assert(other->_basis == _basis &&
        "Should not sum expansions linked against different bases.");
    _coeff = _coeff + other->_coeff;
    return;
}

void PowerExpansion::setCoefficientsNumpy(boost::python::object &np_array) {
    soap::linalg::numpy_converter npc("complex128");
    npc.numpy_to_ublas< std::complex<double> >(np_array, _coeff);
    if (_coeff.size1() != _N*_N ||
        _coeff.size2() != _L+1) {
        throw soap::base::APIError("<PowerExpansion::setCoefficientsNumpy> Matrix size not consistent with basis.");
    }
}

boost::python::object PowerExpansion::getCoefficientsNumpy() {
    soap::linalg::numpy_converter npc("complex128");
    return npc.ublas_to_numpy< std::complex<double> >(_coeff);
}

void PowerExpansion::writeDensity(
    std::string filename,
    Options *options,
    Structure *structure,
    Particle *center) {
    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ofstream::out);
    if (!ofs.is_open()) {
        throw soap::base::IOError("Bad file handle: " + filename);
    }

    double sum_intensity = 0.0;

    PowerExpansion::coeff_t &coeff = this->getCoefficients();
    for (int n = 0; n < _N; ++n) {
        for (int k = 0; k < _N; ++k) {
            for (int l = 0; l <= _L; ++l) {
                std::complex<double> c_nkl = coeff(n*_N+k, l);
                double c_nkl_real = c_nkl.real();
                double c_nkl_imag = c_nkl.imag();
                sum_intensity += c_nkl_real*c_nkl_real + c_nkl_imag*c_nkl_imag;
                ofs << (boost::format("%1$2d %2$2d %3$+2d %4$+1.7e %5$+1.7e") %
                    n % k % l % c_nkl_real % c_nkl_imag) << std::endl;
            }
        }
    }

    GLOG() << "<PowerExpansion::writeDensity> Summed intensity = " << sum_intensity << std::endl;
    ofs.close();
}

void PowerExpansion::registerPython() {
    using namespace boost::python;

    class_<PowerExpansion, PowerExpansion*>("PowerExpansion", init<Basis*>())
        .add_property("array", &PowerExpansion::getCoefficientsNumpy, &PowerExpansion::setCoefficientsNumpy)
        .def("getArray", &PowerExpansion::getCoefficientsNumpy);
}

}

