#include "soap/power.hpp"
#include "soap/linalg/numpy.hpp"

namespace soap {

const std::string PowerExpansion::_numpy_t = "complex128";

PowerExpansion::PowerExpansion(Basis *basis) :
    _basis(basis),
    _L(basis->getAngBasis()->L()),
    _N(basis->getRadBasis()->N()),
    _has_gradients(false) {
    _has_scalars = true;
    _coeff = coeff_zero_t(_N*_N, _L+1);
    // With/without normalization sqrt(8\pi^2/(2l+1)) ?
    _with_sqrt_2l_1_norm = _basis->getOptions()->get<bool>("spectrum.2l1_norm");

}

void PowerExpansion::computeCoefficientsGradients(
        BasisExpansion *basex1, BasisExpansion *basex2, bool same_types) {
    // ATTENTION Order of arguments #1 and #2 matters:
    // o basex1 stores derivative \partial_{i\alpha} Q_{nlm}^\mu(i)
    //   with particle id i, space component alpha, and type mu of particle i
    // o basex2 stores the summed scalar density of particle type nu
    // o same_types is true if mu=nu, false otherwise. This affects
    //   how the derivative of X_{nkl}^{munu} is to be computed

    if (!_basis) throw soap::base::APIError(
        "PowerExpansion::computeCoefficientsGradients, basis not initialised.");

    // ARGUMENTS: DISTINGUISH BETWEEN GRADIENT EXPANSION AND SCALAR EXPANSION
    BasisExpansion *basex_grad;
    BasisExpansion *basex;
    bool grad_first = true;
    if (same_types) {
        basex_grad = basex1;
        basex = basex2;
    }
    else {
        if (basex1->hasGradients() && !basex2->hasGradients()) {
            grad_first = true;
            basex_grad = basex1;
            basex = basex2;
        }
        else if (!basex1->hasGradients() && basex2->hasGradients()) {
            grad_first = false;
            basex_grad = basex2;
            basex = basex1;
        }
        else {
            throw soap::base::APIError(
                "PowerExpansion::computeCoefficientsGradients, arguments not consistent.");
        }
    }

    // SANITY CHECKS
    assert(basex_grad->hasGradients() == true);
    assert(basex->hasGradients() == false);
    assert(basex->hasScalars() == true);

    // Allocate Xnkl components
    _has_gradients = true;
    _coeff_grad_x = coeff_zero_t(_N*_N, _L+1);
    _coeff_grad_y = coeff_zero_t(_N*_N, _L+1);
    _coeff_grad_z = coeff_zero_t(_N*_N, _L+1);

    // Retrieve dQnlm, Qnlm coefficients
    BasisExpansion::coeff_t &dqnlm_dx = basex_grad->getCoefficientsGradX();
    BasisExpansion::coeff_t &dqnlm_dy = basex_grad->getCoefficientsGradY();
    BasisExpansion::coeff_t &dqnlm_dz = basex_grad->getCoefficientsGradZ();
    BasisExpansion::coeff_t &qnlm = basex->getCoefficients();

    // Compute
    if (same_types) {
        for (int n = 0; n < _N; ++n) {
            for (int k = 0; k < _N; ++k) {
                for (int l = 0; l < (_L+1); ++l) {
                    dtype_t c_nkl_dx = 0.0;
                    dtype_t c_nkl_dy = 0.0;
                    dtype_t c_nkl_dz = 0.0;
                    for (int m = -l; m <= l; ++m) {
                        int lm = l*l+l+m;
                        c_nkl_dx += dqnlm_dx(n,lm)*std::conj(qnlm(k,lm))
                                 + qnlm(n,lm)*std::conj(dqnlm_dx(k,lm));
                        c_nkl_dy += dqnlm_dy(n,lm)*std::conj(qnlm(k,lm))
                                 + qnlm(n,lm)*std::conj(dqnlm_dy(k,lm));
                        c_nkl_dz += dqnlm_dz(n,lm)*std::conj(qnlm(k,lm))
                                 + qnlm(n,lm)*std::conj(dqnlm_dz(k,lm));
                    }
                    // Normalization = sqrt(8\pi^2/(2l+1))
                    double prefac_2l_1 = (_with_sqrt_2l_1_norm) ? 2.*sqrt(2.)*M_PI/sqrt(2.*l+1) : 1.; 
                    _coeff_grad_x(n*_N+k, l) = prefac_2l_1*c_nkl_dx;
                    _coeff_grad_y(n*_N+k, l) = prefac_2l_1*c_nkl_dy;
                    _coeff_grad_z(n*_N+k, l) = prefac_2l_1*c_nkl_dz;
                }
            }
        }
    } else {
        for (int n = 0; n < _N; ++n) {
            for (int k = 0; k < _N; ++k) {
                for (int l = 0; l < (_L+1); ++l) {
                    dtype_t c_nkl_dx = 0.0;
                    dtype_t c_nkl_dy = 0.0;
                    dtype_t c_nkl_dz = 0.0;
                    if (grad_first) {
                        for (int m = -l; m <= l; ++m) {
                            int lm = l*l+l+m;
                            c_nkl_dx += dqnlm_dx(n,lm)*std::conj(qnlm(k,lm));
                            c_nkl_dy += dqnlm_dy(n,lm)*std::conj(qnlm(k,lm));
                            c_nkl_dz += dqnlm_dz(n,lm)*std::conj(qnlm(k,lm));
                        }
                    }
                    else {
                        for (int m = -l; m <= l; ++m) {
                            int lm = l*l+l+m;
                            c_nkl_dx += qnlm(n,lm)*std::conj(dqnlm_dx(k,lm));
                            c_nkl_dy += qnlm(n,lm)*std::conj(dqnlm_dy(k,lm));
                            c_nkl_dz += qnlm(n,lm)*std::conj(dqnlm_dz(k,lm));
                        }
                    }
                    // Normalization = sqrt(8\pi^2/(2l+1))
                    double prefac_2l_1 = (_with_sqrt_2l_1_norm) ? 2.*sqrt(2.)*M_PI/sqrt(2.*l+1) : 1.;
                    _coeff_grad_x(n*_N+k, l) = prefac_2l_1*c_nkl_dx;
                    _coeff_grad_y(n*_N+k, l) = prefac_2l_1*c_nkl_dy;
                    _coeff_grad_z(n*_N+k, l) = prefac_2l_1*c_nkl_dz;
                }
            }
        }
    } // not same_types
    return;
}

void PowerExpansion::computeCoefficientsHermConj(
        BasisExpansion *basex1, BasisExpansion *basex2, double scale) {
    if (!_basis) throw soap::base::APIError(
        "PowerExpansion::computeCoefficientsHermConj, basis not initialised.");
    BasisExpansion::coeff_t &coeff1 = basex1->getCoefficients();
    BasisExpansion::coeff_t &coeff2 = basex2->getCoefficients();
    for (int n = 0; n < _N; ++n) {
        for (int k = 0; k < _N; ++k) {
            for (int l = 0; l < (_L+1); ++l) {
                dtype_t c_nkl = 0.0;
                for (int m = -l; m <= l; ++m) {
                    c_nkl += coeff1(n, l*l+l+m)*std::conj(coeff2(k, l*l+l+m));
                }
                double prefac_2l_1 = (_with_sqrt_2l_1_norm) ? 2.*sqrt(2.)*M_PI/sqrt(2.*l+1) : 1.; 
                _coeff(n*_N+k, l) = prefac_2l_1*scale*c_nkl;
            }
        }
    }
    return;
}

void PowerExpansion::computeCoefficients(
        BasisExpansion *basex1, BasisExpansion *basex2) {
    if (!_basis) throw soap::base::APIError(
        "PowerExpansion::computeCoefficients, basis not initialised.");
    BasisExpansion::coeff_t &coeff1 = basex1->getCoefficients();
    BasisExpansion::coeff_t &coeff2 = basex2->getCoefficients();
    for (int n = 0; n < _N; ++n) {
        for (int k = 0; k < _N; ++k) {
            for (int l = 0; l < (_L+1); ++l) {
                dtype_t c_nkl = 0.0;
                for (int m = -l; m <= l; ++m) {
                    c_nkl += coeff1(n, l*l+l+m)*std::conj(coeff2(k, l*l+l+m));
                }
                double prefac_2l_1 = (_with_sqrt_2l_1_norm) ? 2.*sqrt(2.)*M_PI/sqrt(2.*l+1) : 1.;
                _coeff(n*_N+k, l) = prefac_2l_1*c_nkl;
            }
        }
    }
    return;
}

void PowerExpansion::add(PowerExpansion *other) {
    assert(other->_basis == _basis &&
        "Should not sum expansions linked against different bases.");
    _coeff = _coeff + other->_coeff;
    return;
}

void PowerExpansion::setCoefficientsNumpy(boost::python::object &np_array) {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    npc.numpy_to_ublas< dtype_t >(np_array, _coeff);
    if (_coeff.size1() != _N*_N ||
        _coeff.size2() != _L+1) {
        throw soap::base::APIError(
            "<PowerExpansion::setCoefficientsNumpy> Matrix size not consistent with basis.");
    }
}

boost::python::object PowerExpansion::getCoefficientsNumpy() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_coeff);
}

boost::python::object PowerExpansion::getCoefficientsGradXNumpy() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_coeff_grad_x);
}

boost::python::object PowerExpansion::getCoefficientsGradYNumpy() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_coeff_grad_y);
}

boost::python::object PowerExpansion::getCoefficientsGradZNumpy() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_coeff_grad_z);
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
                dtype_t c_nkl = coeff(n*_N+k, l);
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
        .add_property("array", &PowerExpansion::getCoefficientsNumpy, 
            &PowerExpansion::setCoefficientsNumpy)
        .def("getArray", &PowerExpansion::getCoefficientsNumpy)
        .def("getArrayGradX", &PowerExpansion::getCoefficientsGradXNumpy)
        .def("getArrayGradY", &PowerExpansion::getCoefficientsGradYNumpy)
        .def("getArrayGradZ", &PowerExpansion::getCoefficientsGradZNumpy);
}

}
