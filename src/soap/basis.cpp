#include <math.h>

#include "soap/linalg/numpy.hpp"
#include "soap/basis.hpp"


namespace soap {

// =====
// Basis
// =====

Basis::Basis(Options *options) :
    _options(options) {
    GLOG() << "Configuring basis ..." << std::endl;
    // CONFIGURE RADIAL BASIS
    _radbasis = RadialBasisOutlet().create(_options->get<std::string>("radialbasis.type"));
    _radbasis->configure(*options);
    // CONFIGURE ANGULAR BASIS
    _angbasis = AngularBasisOutlet().create(_options->get<std::string>("angularbasis.type"));
    _angbasis->configure(*options);
    // CONFIGURE CUTOFF FUNCTION
    _cutoff = CutoffFunctionOutlet().create(_options->get<std::string>("radialcutoff.type"));
    _cutoff->configure(*options);
}

Basis::~Basis() {
    delete _radbasis;
    _radbasis = NULL;
    delete _angbasis;
    _angbasis = NULL;
    delete _cutoff;
    _cutoff = NULL;
}

void Basis::registerPython() {
    using namespace boost::python;
    class_<Basis, Basis*>("Basis", init<>())
        .def(init<Options*>())
        .add_property("N", make_function(&Basis::N, copy_const()))
        .add_property("L", make_function(&Basis::L, copy_const()));
}

// ==============
// BasisExpansion
// ==============

BasisExpansion::BasisExpansion(Basis *basis) :
    _basis(basis),
    _radbasis(basis->getRadBasis()),
    _angbasis(basis->getAngBasis()),
    _has_scalars(false),
    _has_gradients(false) {
    int L = _angbasis->L();
    int N = _radbasis->N();
    // ZERO SCALAR-FIELD CONTAINERS
    _has_scalars = true;
    _radcoeff = RadialBasis::radcoeff_zero_t(N,L+1);
    _angcoeff = AngularBasis::angcoeff_zero_t((L+1)*(L+1));
    _coeff = coeff_zero_t(N,(L+1)*(L+1));
}

BasisExpansion::~BasisExpansion() {
    _basis = NULL;
    _radbasis = NULL;
    _angbasis = NULL;
    _radcoeff.clear();
    _angcoeff.clear();
}

void BasisExpansion::computeCoefficients(double r, vec d) {
    // Corresponds to: weight=weight_scale=1, sigma=0, gradients=false
    _radbasis->computeCoefficients(d, r, 0., _radcoeff, NULL, NULL, NULL);
    _angbasis->computeCoefficients(d, r, 0., _angcoeff, NULL, NULL, NULL);
    for (int n = 0; n != _radbasis->N(); ++n) {
        for (int l = 0; l != _angbasis->L()+1; ++l) {
            for (int m = -l; m != l+1; ++m) {
                _coeff(n, l*l+l+m) = _radcoeff(n,l)*_angcoeff(l*l+l+m);
            }
        }
    }
    return;
}

void BasisExpansion::computeCoefficients(double r, vec d, double weight, double weight_scale, double sigma, bool gradients) {
    // SETUP STORAGE
    int L = _angbasis->L();
    int N = _radbasis->N();
    if (!_has_scalars) {
        // Should have already been done in constructor ::BasisExpansion(Basis *basis)
        _has_scalars = true;
        _radcoeff = RadialBasis::radcoeff_zero_t(N,L+1);
        _angcoeff = AngularBasis::angcoeff_zero_t((L+1)*(L+1));
        _coeff = coeff_zero_t(N,(L+1)*(L+1));
    }
    if (gradients) {
        _has_gradients = true;
        _radcoeff_grad_x = RadialBasis::radcoeff_zero_t(N,L+1);
        _radcoeff_grad_y = RadialBasis::radcoeff_zero_t(N,L+1);
        _radcoeff_grad_z = RadialBasis::radcoeff_zero_t(N,L+1);
        _angcoeff_grad_x = AngularBasis::angcoeff_zero_t((L+1)*(L+1));
        _angcoeff_grad_y = AngularBasis::angcoeff_zero_t((L+1)*(L+1));
        _angcoeff_grad_z = AngularBasis::angcoeff_zero_t((L+1)*(L+1));
        _coeff_grad_x = coeff_zero_t(N,(L+1)*(L+1));
        _coeff_grad_y = coeff_zero_t(N,(L+1)*(L+1));
        _coeff_grad_z = coeff_zero_t(N,(L+1)*(L+1));
    }
    // COMPUTE
    if (_has_scalars && !_has_gradients) {
        _radbasis->computeCoefficients(d, r, sigma, _radcoeff, NULL, NULL, NULL);
        _angbasis->computeCoefficients(d, r, sigma, _angcoeff, NULL, NULL, NULL);
    }
    else if (_has_scalars && _has_gradients) {
        //std::cout << "GRAD" << std::endl;
        _radbasis->computeCoefficients(d, r, sigma, _radcoeff, &_radcoeff_grad_x, &_radcoeff_grad_y, &_radcoeff_grad_z);
        _angbasis->computeCoefficients(d, r, sigma, _angcoeff, &_angcoeff_grad_x, &_angcoeff_grad_y, &_angcoeff_grad_z);
        _weight_scale_grad = _basis->getCutoff()->calculateGradientWeight(r, d);
    }
    // MERGE
    if (_has_scalars) {
        for (int n = 0; n != _radbasis->N(); ++n) {
            for (int l = 0; l != _angbasis->L()+1; ++l) {
                for (int m = -l; m != l+1; ++m) {
                    _coeff(n, l*l+l+m) = _radcoeff(n,l)*_angcoeff(l*l+l+m);
                }
            }
        }
        _coeff *= weight*weight_scale;
    }
    if (_has_gradients) {
        for (int n = 0; n != _radbasis->N(); ++n) {
            for (int l = 0; l != _angbasis->L()+1; ++l) {
                for (int m = -l; m != l+1; ++m) {
                    int lm = l*l+l+m;
                    _coeff_grad_x(n, lm) = weight*(
                          weight_scale*_radcoeff(n,l)*_angcoeff_grad_x(lm)
                        + weight_scale*_radcoeff_grad_x(n,l)*_angcoeff(lm)
                        + _weight_scale_grad.getX()*_radcoeff(n,l)*_angcoeff(lm)
                    );
                    _coeff_grad_y(n, lm) = weight*(
                          weight_scale*_radcoeff(n,l)*_angcoeff_grad_y(lm)
                        + weight_scale*_radcoeff_grad_y(n,l)*_angcoeff(lm)
                        + _weight_scale_grad.getY()*_radcoeff(n,l)*_angcoeff(lm)
                    );
                    _coeff_grad_z(n, lm) = weight*(
                          weight_scale*_radcoeff(n,l)*_angcoeff_grad_z(lm)
                        + weight_scale*_radcoeff_grad_z(n,l)*_angcoeff(lm)
                        + _weight_scale_grad.getZ()*_radcoeff(n,l)*_angcoeff(lm)
                    );
                } // m
            } // l
        } // n
    }

    // CLEAR INTERMEDIATE STORAGE (NOT REQUIRED LATER)
    _radcoeff.clear();
    _angcoeff.clear();
    _radcoeff_grad_x.clear();
    _radcoeff_grad_y.clear();
    _radcoeff_grad_z.clear();
    _angcoeff_grad_x.clear();
    _angcoeff_grad_y.clear();
    _angcoeff_grad_z.clear();

    return;
}

void BasisExpansion::addGradient(BasisExpansion &other) {
    assert(_has_gradients);
    _coeff_grad_x = _coeff_grad_x + other.getCoefficientsGradX();
    _coeff_grad_y = _coeff_grad_y + other.getCoefficientsGradY();
    _coeff_grad_z = _coeff_grad_z + other.getCoefficientsGradZ();
    return;
}

void BasisExpansion::zeroGradient() {
    _has_gradients = true;
    int L = _angbasis->L();
    int N = _radbasis->N();
    _coeff_grad_x = coeff_zero_t(N,(L+1)*(L+1));
    _coeff_grad_y = coeff_zero_t(N,(L+1)*(L+1));
    _coeff_grad_z = coeff_zero_t(N,(L+1)*(L+1));
    return;
}

void BasisExpansion::conjugate() {
    for (int n = 0; n != _radbasis->N(); ++n) {
        for (int l = 0; l != _angbasis->L()+1; ++l) {
            for (int m = -l; m != l+1; ++m) {
                _coeff(n, l*l+l+m) = std::conj(_coeff(n, l*l+l+m));
            }
        }
    }
}

void BasisExpansion::writeDensity(
    std::string filename,
    Options *options,
    Structure *structure,
    Particle *center) {
    if (_angbasis == NULL || _radbasis == NULL) {
        throw soap::base::APIError("<BasisExpansion::writeDensityOnGrid> "
            "Object not linked against basis.");
    }

    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ofstream::out);
    if (!ofs.is_open()) {
        throw soap::base::IOError("Bad file handle: " + filename);
    }

    double sum_intensity = 0.0;

    BasisExpansion::coeff_t &coeff = this->getCoefficients();
    for (int n = 0; n < _radbasis->N(); ++n) {
        for (int l = 0; l <= _angbasis->L(); ++l) {
            for (int m = -l; m <= l; ++m) {
                std::complex<double> c_nlm = coeff(n, l*l+l+m);
                double c_nlm_real = c_nlm.real();
                double c_nlm_imag = c_nlm.imag();
                sum_intensity += c_nlm_real*c_nlm_real + c_nlm_imag*c_nlm_imag;
                ofs << (boost::format("n %1$2d l %2$2d m %3$+2d %4$+1.7e %5$+1.7e") %
                    n % l %m % c_nlm_real % c_nlm_imag) << std::endl;
            }
        }
    }

    GLOG() << "<BasisExpansion::writeDensity> Summed intensity = " << sum_intensity << std::endl;
    ofs.close();
}

void BasisExpansion::writeDensityOnGrid(
    std::string filename,
    Options *options,
    Structure *structure,
    Particle *center,
    bool fromExpansion) {

    if (_angbasis == NULL || _radbasis == NULL) {
        throw soap::base::APIError("<BasisExpansion::writeDensityOnGrid> "
            "Object not linked against basis.");
    }

    int I = options->get<int>("densitygrid.N");
    int Nx = 2*I+1;
    int Ny = Nx;
    int Nz = Nx;

    double dx = options->get<double>("densitygrid.dx");
    double dy = dx;
    double dz = dx;

    double conv = soap::constants::ANGSTROM_TO_BOHR;

    vec r0 = -I * vec(dx,dy,dz);

    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ofstream::out);
    if (!ofs.is_open()) {
        throw soap::base::IOError("Bad file handle: " + filename);
    }

    ofs << "DENSITY ON GRID" << std::endl;
    ofs << "Generated by BasisCoefficients::writeDensityOnGrid" << std::endl;
    ofs << boost::format("%1$d %2$+1.4f %3$+1.4f %4$+1.4f")
        % structure->particles().size() % (r0.x()*conv) % (r0.y()*conv) % (r0.z()*conv) << std::endl;
    ofs << boost::format("%1$d %2$+1.4f +0.0000 +0.0000") % Nx % (dx*conv) << std::endl;
    ofs << boost::format("%1$d +0.0000 %2$+1.4f +0.0000") % Ny % (dy*conv) << std::endl;
    ofs << boost::format("%1$d +0.0000 +0.0000 %2$+1.4f") % Nz % (dz*conv) << std::endl;

    Structure::particle_it_t pit;
    for (pit = structure->beginParticles(); pit != structure->endParticles(); ++pit) {
         vec dr = structure->connect(center->getPos(), (*pit)->getPos());
         ofs << boost::format("%1$d 0.0 %2$+1.4f %3$+1.4f %4$+1.4f\n")
             % (*pit)->getTypeId() % (dr.x()*conv) % (dr.y()*conv) % (dr.z()*conv);
    }

    GLOG() << "'" << filename << "': " << "Fill grid " << std::flush;
    double int_density_dr = 0.0;
    int ijk = 0;
    for (int i = -I; i <= I; ++i) {
        if (((i+I) % 5) == 0) GLOG() << "." << std::flush;
        for (int j = -I; j <= I; ++j) {
            for (int k = -I; k <= I; ++k) {
                double density_dr = 0.0;

                // Note that expansion is computed with respect to center-particle position:
                // Hence no shifting required
                vec dr(i*dx, j*dy, k*dz);
                double r = soap::linalg::abs(dr);
                vec d = dr/r;

                // DENSITY BASED ON EXPANSION
                if (fromExpansion) {
                    BasisExpansion density_exp_dr(_basis);
                    //density_exp_dr.computeCoefficients(r, d, 1., 1., 0., false);
                    density_exp_dr.computeCoefficients(r, d);
                    density_exp_dr.conjugate();

                    BasisExpansion::coeff_t &c_nlm_dr = density_exp_dr.getCoefficients();
                    BasisExpansion::coeff_t &c_nlm = this->getCoefficients();
                    // Something like this would be desirable to replace n-l-m loop below:
                    //double density_dr = ub::inner_prod(c_nlm, c_nlm_dr);
                    for (int n = 0; n < _radbasis->N(); ++n) {
                        for (int l = 0; l <= _angbasis->L(); ++l) {
                            for (int m = -l; m <= l; ++m) {
                                double voxel_density_dr = (c_nlm(n, l*l+l+m)*c_nlm_dr(n, l*l+l+m)).real();
                                density_dr += voxel_density_dr;
                                int_density_dr += voxel_density_dr*dx*dy*dz;
                            }
                        }
                    }
                }

                // DENSITY BASED ON SMEARED PARTICLES
                else {
                    for (pit = structure->beginParticles(); pit != structure->endParticles(); ++pit) {
                         vec dr_center_particle = structure->connect(center->getPos(), (*pit)->getPos());
                         vec dr_particle_target = dr - dr_center_particle;
                         double r_particle_target = soap::linalg::abs(dr_particle_target);
                         double sigma = (*pit)->getSigma();
                         double weight = (*pit)->getWeight();
                         if (r_particle_target > 4.*sigma) continue;
                         density_dr += weight*pow(1./(2.*M_PI*sigma*sigma), 1.5)
                             * exp(-r_particle_target*r_particle_target/(2*sigma*sigma));
                    }
                    int_density_dr += density_dr*dx*dy*dz;
                }

                ofs << density_dr << " ";
                ijk += 1;
                ijk = ijk % 6;
                if (ijk == 0) {
                    ofs << std::endl;
                }

            }
        }
    }
    ofs.close();
    GLOG() << " Volume integral = " << int_density_dr << std::endl;
    return;
}

void BasisExpansion::setCoefficientsNumpy(boost::python::object &np_array) {
    soap::linalg::numpy_converter npc("complex128");
    npc.numpy_to_ublas< std::complex<double> >(np_array, _coeff);
    int N = _basis->getRadBasis()->N();
    int L = _basis->getAngBasis()->L();
    if (_coeff.size1() != N ||
        _coeff.size2() != (L+1)*(L+1)) {
        throw soap::base::APIError("<BasisExpansion::setCoefficientsNumpy> Matrix size not consistent with basis.");
    }
}

boost::python::object BasisExpansion::getCoefficientsNumpy() {
    soap::linalg::numpy_converter npc("complex128");
    return npc.ublas_to_numpy< std::complex<double> >(_coeff);
}

void BasisExpansion::registerPython() {
    using namespace boost::python;

    class_<BasisExpansion, BasisExpansion*>("BasisExpansion", init<Basis*>())
        .add_property("array", &BasisExpansion::getCoefficientsNumpy, &BasisExpansion::setCoefficientsNumpy)
        .def("getArray", &BasisExpansion::getCoefficientsNumpy)
        .def("writeDensity", &BasisExpansion::writeDensity)
        .def("writeDensityOnGrid", &BasisExpansion::writeDensityOnGrid);
}

}

