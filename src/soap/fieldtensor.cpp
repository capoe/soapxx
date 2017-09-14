#include "soap/fieldtensor.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/wigner.hpp"
#include <boost/math/special_functions/legendre.hpp>

namespace soap {

namespace ub = boost::numeric::ublas;

// ============================
// Spectrum::FTSpectrum::Atomic
// ============================

const std::string AtomicSpectrumFT::_numpy_t = "float64";

AtomicSpectrumFT::AtomicSpectrumFT(Particle *center, int K, int L, Options *options)
    : _center(center), _K(K), _L(L) {
    this->_s = center->getTypeId()-1; // TODO No longer used, using string labels as map keys instead
    this->_type = center->getType();
    if (GLOG.verbose())
        GLOG() << "Created FT particle: " << _type
            << " " << center->getId()
            << " @ " << _center->getPos()
            << std::endl;
    //assert(_s >= 0 && "Type-IDs should start from 1");
    // Body-order storage
    assert(K >= 0);
    _body_map.resize(K+1);
    // Linear field responses
    double p = options->get<double>("fieldtensor.alpha.p");
    double q = options->get<double>("fieldtensor.alpha.q");
    double r = options->get<double>("fieldtensor.alpha.r");
    _alpha.clear();
    for (int l = 0; l <= _L; ++l) {
        //double alpha_l = factorial(l)*pow(2*l+1,0.0)*pow(_center->getSigma(), 2*l+1); // HACK
        //
        //double alpha_l = pow(2*l+1,1.0)*pow(_center->getSigma(), 2*l+1);
        //if (l == 0) alpha_l *= 0.5;
        //
        //double alpha_l = 1.;
        //
        double alpha_l = pow(2*l+1,p)*pow(_center->getSigma(), q*(2*l+1));
        if (l == 0) alpha_l *= r;
        _alpha.push_back(alpha_l);
    }
}

AtomicSpectrumFT::~AtomicSpectrumFT() {
    #ifdef DBG
        GLOG() << "[~] Destruct " << this->getCenter()->getId() << std::endl;
    #endif
    // Deallocate body-order field terms
    for (int k=0; k <= _K; ++k) {
        field_map_t &fm = _body_map[k];
        #ifdef DBG
            GLOG() << "[~]   Deallocating k=" << k<< std::endl;
        #endif
        // Deallocate field terms for each type channel
        for (auto it=fm.begin(); it != fm.end(); ++it) {
            #ifdef DBG
                GLOG() << "[~]     Deallocating type=" <<  it->first << std::endl;
            #endif
            delete it->second;
        }
    }
    // Deallocate contraction coefficients
    for (auto it = _coeff_map.begin(); it != _coeff_map.end(); ++it) {
        #ifdef DBG
            GLOG() << "[~]   Deallocating s1:s2 = "
                <<  it->first.first << ":" << it->first.second
                << std::endl;
        #endif
        delete it->second;
    }
}

boost::python::list AtomicSpectrumFT::getTypes() {
    boost::python::list types;
    // The highest-body-order term should have the complete set
    for (auto it = _body_map[_K].begin(); it != _body_map[_K].end(); ++it) {
        std::string type = it->first;
        types.append(type);
    }
    return types;
}

boost::python::object AtomicSpectrumFT::getCoefficientsNumpy(std::string s1, std::string s2) {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    channel_t channel(s1, s2);
    return npc.ublas_to_numpy< dtype_t >(*_coeff_map[channel]);
}

void AtomicSpectrumFT::addField(int k, std::string type, field_t &flm) {
    field_t &f = *(this->getCreateField(k, type, flm.size1(), flm.size2()));
    f = f + flm;
    return;
}

AtomicSpectrumFT::field_t *AtomicSpectrumFT::getCreateField(int k, std::string type, int s1, int s2) {
    assert(k <= _K);
    auto it = _body_map[k].find(type);
    if (it == _body_map[k].end()) {
        #ifdef DBG
            GLOG() << "[AtomicSpectrumFT::getCreateField]"
                << " Allocate k " << k
                << " type '" << type
                << "' [" << s1 << "x" << s2 << "]"
                << std::endl;
        #endif
        _body_map[k][type] = new field_t(
            field_zero_t(s1, s2)
        );
        it = _body_map[k].find(type);
    }
    return it->second;
}

AtomicSpectrumFT::coeff_t *AtomicSpectrumFT::getCreateContraction(channel_t &channel, int size1, int size2) {
    // size1 -> trail dimension lambda_total
    // size2 -> L+1
    auto it = _coeff_map.find(channel);
    if (it == _coeff_map.end()) {
        #ifdef DBG
            GLOG() << "[AtomicSpectrumFT::getCreateContraction]"
                << " Allocate " << channel.first << ":" << channel.second
                << " " << size1 << "x" << size2
                << std::endl;
        #endif
        _coeff_map[channel] = new coeff_t(
            coeff_zero_t(size1, size2)
        );
        it = _coeff_map.find(channel);
    }
    return it->second;
}

AtomicSpectrumFT::field_t *AtomicSpectrumFT::getField(int k, std::string type) {
    assert(k <= _K);
    auto it = _body_map[k].find(type);
    if (it == _body_map[k].end()) {
        return NULL;
    }
    else {
        return it->second;
    }
}

void AtomicSpectrumFT::contractDeep() {
    /*
    for (int l1=0; l1<=_L; ++l1) {
    for (int l2=0; l2<=_L; ++l2) {
    for (int l3=0; l3<=_L; ++l3) {
       int m1 = -l1;
       int m2 = -l2;
       int m3 = -m1-m2;
       double w = WignerSymbols::wigner3j(l1, l2, l3, m1, m2, m3);
       GLOG() << l1 << ":" << l2 << ":" << l3 << " " << m1 << ":" << m2 << ":" << m3 << " " << w << std::endl;
    }}}

    for (int l1=0; l1<=_L; ++l1) {
    for (int l2=0; l2<=_L; ++l2) {
    for (int l3=0; l3<=0; ++l3) {
        for (int m1=-l1; m1<=l1; ++m1) {
            int m2 = -m1;
            int m3 = 0;
            double w = WignerSymbols::wigner3j(l1, l2, l3, m1, m2, m3);
            GLOG() << l1 << ":" << l2 << ":" << l3 << " " << m1 << ":" << m2 << ":" << m3 << " " << w << std::endl;
        }
    }}}
    */

    int k = 1;
    int size_l1l1 = _L+1;
    int size_l1l1l1 = _L+1;
    // Unfortunately, there is no analytical formula for the descriptor size.
    // Hence tabulation based on <l1,l2,l3> loops below.
    // Wig. 3j(2)                _L =  0    1    2    3    4    5   6    7    8    9    10   11   12   13   14   15
    std::vector<int> size_l1l1l2 = {   0,   1,   4,   8,  14,  21,  30,  40,  52,  65,  80,  96, 114, 133, 154, 176 };
    std::vector<int> size_l1l2l3 = {   0,   0,   0,   1,   3,   7,  13,  22,  34,  50,  70,  95, 125, 161, 203, 252 };
    int length = size_l1l1 + size_l1l1l1 + 2*size_l1l1l2[_L] + 3*size_l1l2l3[_L];
    field_map_t &field_map = _body_map[k];

    // Contract channels
    for (auto it1 = field_map.begin(); it1 != field_map.end(); ++it1) {
        std::string s1 = it1->first; // <- Channel 1 (e.g., 'C')
        field_t &f1 = *(it1->second);
        for (auto it2 = field_map.begin(); it2 != field_map.end(); ++it2) {
            std::string s2 = it2->first; // <- Channel 2 (e.g., 'O')
            field_t &f2 = *(it2->second);
            GLOG() << " " << s1 << ":" << s2 << std::flush;

            // Allocate channel if necessary
            channel_t channel(s1, s2);
            coeff_t &coeffs = *(this->getCreateContraction(channel, length, 1)); // TODO Will not work with k-dependent L-cutoff

            // a,l <> a',l contractions
            int offset = 0;
            for (int l = 0; l <= _L; ++l) {
                double inv_alpha_l = 1./this->_alpha[l];
                cmplx_t phi_l_s1s2 = 0.0;
                for (int m = -l; m <= l; ++m) {
                    int lm = l*l+l+m;
                    phi_l_s1s2 += f1(0, lm)*std::conj(f2(0, lm));
                }
                #ifdef DBG
                    GLOG() << l << ":" << l << std::endl;
                    GLOG() << "Save @ " << offset << std::endl;
                #endif
                coeffs(offset, 0) = pow(this->getCenter()->getSigma(), 2*l)*inv_alpha_l*phi_l_s1s2.real();
                offset += 1;
            } // l

            // al, a'l, a'l contractions
            for (int l1 = 0; l1 <= _L; ++l1) {
                double inv_alpha_l1 = 1./this->_alpha[l1];
                cmplx_t phi_l1_l1l1 = 0.0;
                double w111_sq = 0.0;
                for (int m1 = -l1; m1 <= l1; ++m1) {
                    int lm1 = l1*l1+l1+m1;
                for (int m2 = -l1; m2 <= l1; ++m2) {
                    int lm2 = l1*l1+l1+m2;
                for (int m3 = -l1; m3 <= l1; ++m3) {
                    int lm3 = l1*l1+l1+m3;
                    double w111 = WignerSymbols::wigner3j(l1, l1, l1, m1, m2, m3);
                    w111_sq += w111*w111;
                    phi_l1_l1l1 += w111 * f1(0,lm1) * f2(0,lm2) * f2(0,lm3);
                }}}
                // Store l1l1l2_s1s1s2
                #ifdef DBG
                    GLOG() << l1 << "::" << l1 << ":" << l1 << " " << w111_sq << std::endl;
                    GLOG() << "Save @ " << offset << std::endl;
                #endif
                coeffs(offset, 0) = 
                        sqrt(inv_alpha_l1)*sqrt(inv_alpha_l1)*sqrt(inv_alpha_l1)
                      * pow(this->getCenter()->getSigma(), l1+l1+l1+0.5) * 
                        pow(-1, l1-l1+l1)
                      * phi_l1_l1l1.real();
                offset += 1;
            }

            // al, a'l', a'l' contractions
            for (int l1 = 0; l1 <= _L; ++l1) {
                double inv_alpha_l1 = 1./this->_alpha[l1];
            for (int l2 = 0; l2 <= _L && l2 <= 2*l1; ++l2) {
                // Redundancy check
                if (l1 == l2) continue; // Dealt with above
                double inv_alpha_l2 = 1./this->_alpha[l2];
                // Contract: (al1 al1 bl2) or (al1 bl1 bl2)
                cmplx_t phi_l1l1_l2 = 0.0;
                cmplx_t phi_l1_l1l2 = 0.0;
                double w112_sq = 0.0;
                for (int m1 = -l1; m1 <= l1; ++m1) {
                    int lm1 = l1*l1+l1+m1;
                for (int m2 = -l1; m2 <= l1; ++m2) {
                    int lm2 = l1*l1+l1+m2;
                for (int m3 = -l2; m3 <= l2; ++m3) {
                    int lm3 = l2*l2+l2+m3;
                    //double w122 = WignerSymbols::wigner3j(l2, l1, l2, m2, m1, m3);
                    double w112 = WignerSymbols::wigner3j(l1, l1, l2, m1, m2, m3);
                    w112_sq += w112*w112;
                    //GLOG() << l1 << ":" << l2 << ":" << m1 << ":" << m2 << "  " << w122 << std::endl;
                    //phi_l1l2l2_s1s2 += w122 * f1(0,lm2) * f2(0,lm1) * f2(0,lm3);
                    phi_l1l1_l2 += w112 * f1(0,lm1) * f1(0,lm2) * f2(0,lm3);
                    phi_l1_l1l2 += w112 * f1(0,lm1) * f2(0,lm2) * f2(0,lm3);
                }}}
                // Store l1l1l2_s1s1s2
                #ifdef DBG
                    GLOG() << l1 << ":" << l1 << "::" << l2 << " " << w112_sq << std::endl;
                    GLOG() << "Save @ " << offset << std::endl;
                #endif
                coeffs(offset, 0) = 
                        sqrt(inv_alpha_l1)*sqrt(inv_alpha_l1)*sqrt(inv_alpha_l2)
                      * pow(this->getCenter()->getSigma(), l1+l1+l2+0.5) * 
                        pow(-1, l1-l1+l2)
                      * phi_l1l1_l2.real();
                offset += 1;
                // Store l1l1l2_s1s2s2
                #ifdef DBG
                    GLOG() << l1 << "::" << l1 << ":" << l2 << " " << w112_sq << std::endl;
                    GLOG() << "Save @ " << offset << std::endl;
                #endif
                coeffs(offset, 0) = 
                        sqrt(inv_alpha_l1)*sqrt(inv_alpha_l1)*sqrt(inv_alpha_l2)
                      * pow(this->getCenter()->getSigma(), l1+l1+l2+0.5) * 
                        pow(-1, l1-l1+l2)
                      * phi_l1_l1l2.real();
                offset += 1;
            }}

            // al, a'l', a'l'' contractions
            for (int l1 = 0; l1 <= _L; ++l1) {
                double inv_alpha_l1 = 1./this->_alpha[l1];
            for (int l2 = l1+1; l2 <= _L; ++l2) {
                double inv_alpha_l2 = 1./this->_alpha[l2];
            for (int l3 = l2+1; l3 <= l2+l1 && l3 <= _L; ++l3) {
                double inv_alpha_l3 = 1./this->_alpha[l3];
                cmplx_t phi_l1_l2l3 = 0.0;
                cmplx_t phi_l2_l3l1 = 0.0;
                cmplx_t phi_l3_l1l2 = 0.0;
                double w123_sq = 0.0;
                for (int m1 = -l1; m1 <= l1; ++m1) {
                    int lm1 = l1*l1+l1+m1;
                for (int m2 = -l2; m2 <= l2; ++m2) {
                    int lm2 = l2*l2+l2+m2;
                for (int m3 = -l3; m3 <= l3; ++m3) {
                    int lm3 = l3*l3+l3+m3;
                    double w123 = WignerSymbols::wigner3j(l1, l2, l3, m1, m2, m3);
                    w123_sq += w123*w123;
                    phi_l1_l2l3 += w123 * f1(0,lm1) * f2(0,lm2) * f2(0,lm3);
                    phi_l2_l3l1 += w123 * f1(0,lm2) * f2(0,lm3) * f2(0,lm1);
                    phi_l3_l1l2 += w123 * f1(0,lm3) * f2(0,lm1) * f2(0,lm2);
                }}}
                // Store (l1)(l2l3)
                #ifdef DBG
                    GLOG() << l1 << "::" << l2 << ":" << l3 << " " << w123_sq << std::endl;
                    GLOG() << "Save @ " << offset << std::endl;
                #endif
                coeffs(offset, 0) = 
                        sqrt(inv_alpha_l1)*sqrt(inv_alpha_l2)*sqrt(inv_alpha_l3)
                      * pow(this->getCenter()->getSigma(), l1+l2+l3+0.5) * 
                        pow(-1, l1-l2+l3)
                      * phi_l1_l2l3.real();
                offset += 1;
                // Store (l2)(l3l1)
                #ifdef DBG
                    GLOG() << l2 << "::" << l3 << ":" << l1 << " " << w123_sq << std::endl;
                    GLOG() << "Save @ " << offset << std::endl;
                #endif
                coeffs(offset, 0) = 
                        sqrt(inv_alpha_l1)*sqrt(inv_alpha_l2)*sqrt(inv_alpha_l3)
                      * pow(this->getCenter()->getSigma(), l1+l2+l3+0.5) * 
                        pow(-1, l1-l2+l3)
                      * phi_l2_l3l1.real();
                offset += 1;
                // Store (l3)(l1l2)
                #ifdef DBG
                    GLOG() << l3 << "::" << l1 << ":" << l2 << " " << w123_sq << std::endl;
                    GLOG() << "Save @ " << offset << std::endl;
                #endif
                coeffs(offset, 0) = 
                        sqrt(inv_alpha_l1)*sqrt(inv_alpha_l2)*sqrt(inv_alpha_l3)
                      * pow(this->getCenter()->getSigma(), l1+l2+l3+0.5) * 
                        pow(-1, l1-l2+l3)
                      * phi_l3_l1l2.real();
                offset += 1;
            }}}


        } // Channel type 2
    } // Channel type 1
    GLOG() << std::endl;

    return;
}

void AtomicSpectrumFT::contract() {
    int k = 0;
    int Lambda_total = (1 - pow(_L+1, _K))/(1 - (_L+1)); // from geometric series
    int LambdaLambda_total = (1 - pow(_L+1, 2*_K))/(1 - pow(_L+1, 2));

    GLOG() << "[contract] Centre: " << this->getCenter()->getId() << ":" << this->getType() << std::flush;
    #ifdef DBG
        GLOG() << std::endl;
        GLOG() << "[contract] Trail length total: " << Lambda_total << std::endl;
        GLOG() << "[contract] Trail length total - contracted: " << LambdaLambda_total << std::endl;
    #endif
    for (k=1; k<=_K; ++k) { // TODO Mix different k-channels? That would increase descriptor length if affordable/required.

        int Lambda_off_k = (1 - pow(_L+1, k-1))/(1 - (_L+1));
        int LambdaLambda_off_k = (1 - pow(_L+1, 2*(k-1)))/(1 - pow(_L+1, 2));
        int Lambda_k = pow(_L+1, k-1);
        int L_k = _L;

        #ifdef DBG
            GLOG()
                << "[contract] L_k= " << _L
                << " Lambda_k= " << Lambda_k
                << " Lambda_off_k= " << Lambda_off_k
                << " LambdaLambda_off_k= " << LambdaLambda_off_k
                << std::endl;
        #endif

        field_map_t &field_map = _body_map[k];
        #ifdef DBG
            for (auto it = field_map.begin(); it != field_map.end(); ++it) {
                std::string type = it->first;
                field_t &field = *(it->second);
                GLOG() << "[contract]  k= " << k
                    << " type= " << type
                    << " ||lambda||= " << field.size1()
                    << " ||L||= " << field.size2()
                    << std::endl;
            }
        #endif
        // Contract channels
        for (auto it1 = field_map.begin(); it1 != field_map.end(); ++it1) {
            std::string s1 = it1->first; // <- Channel 1 (e.g., 'C')
            field_t &f1 = *(it1->second);
            for (auto it2 = field_map.begin(); it2 != field_map.end(); ++it2) {
                std::string s2 = it2->first; // <- Channel 2 (e.g., 'O')
                field_t &f2 = *(it2->second);
                GLOG() << " " << s1 << ":" << s2 << std::flush;
                // Size sanity checks
                int Lambda1 = f1.size1();
                int Lambda2 = f2.size1();
                int L1 = f1.size2();
                int L2 = f2.size2();
                assert(L1 == (_L+1)*(_L+1));
                assert(L2 == (_L+1)*(_L+1));
                assert(Lambda1 == Lambda_k);
                assert(Lambda2 == Lambda_k);
                // Allocate channel if necessary
                channel_t channel(s1, s2);
                coeff_t &coeffs = *(this->getCreateContraction(channel, LambdaLambda_total, _L+1)); // TODO Will not work with k-dependent L-cutoff
                #ifdef DBG
                    GLOG() << "Coefficient matrix for this channel: "
                        << coeffs.size1() << "x" << coeffs.size2()
                        << std::endl;
                #endif
                for (int lambda1 = 0; lambda1 < Lambda1; ++lambda1) {
                    for (int lambda2 = 0; lambda2 < Lambda2; ++lambda2) {
                        for (int l = 0; l <= L_k; ++l) {
                            // Polarizability
                            //double inv_alpha = 1./(2*l+1)*pow(this->getCenter()->getSigma(), -2*l-1); // TODO Initialise in constructor HACK 1./(2*l+1)
                            //double inv_alpha = 1./(2*l+1)*pow(this->getCenter()->getSigma(), -l);
                            //double inv_alpha = 1./(l+0.1)*pow(this->getCenter()->getSigma(), -l);
                            double inv_alpha = 1./this->_alpha[l];
                            cmplx_t phi_l_s1s2_l1l2 = 0.0;
                            for (int m = -l; m <= l; ++m) {
                                int lm = l*l+l+m;
                                phi_l_s1s2_l1l2 += f1(lambda1, lm)*std::conj(f2(lambda2, lm));
                            }
                            int l1l2 = LambdaLambda_off_k+lambda1*Lambda1 + lambda2;
                            #ifdef DBG
                                GLOG() << " Store "
                                    << lambda1 << ":" << lambda2 << ":" << l
                                    << " @ " << l1l2 << ":" << l
                                    << " = " << phi_l_s1s2_l1l2
                                    <<  std::endl;
                            #endif
                            //coeffs(l1l2, l) = inv_alpha*phi_l_s1s2_l1l2.real(); // TODO HACK
                            coeffs(l1l2, l) = pow(this->getCenter()->getSigma(), 2*l)*inv_alpha*phi_l_s1s2_l1l2.real(); // TODO HACK
                            //coeffs(l1l2, l) = phi_l_s1s2_l1l2.real(); // TODO HACK

                        } // l
                    } // lambda 2
                } // lambda 1
            } // Channel type 2
        } // Channel type 1
    }
    GLOG() << std::endl;
    return;
}

void AtomicSpectrumFT::registerPython() {
    using namespace boost::python;
    class_<AtomicSpectrumFT, AtomicSpectrumFT*>("AtomicSpectrumFT", init<Particle*, int, int, Options*>())
        .def("getTypes", &AtomicSpectrumFT::getTypes)
        .def("getPower", &AtomicSpectrumFT::getCoefficientsNumpy)
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
        for (int m = -l; m <= l; ++m) {
            out[l*l+l+m] = sqrt(factorial(l+m)*factorial(l-m));
            #ifdef DBG // TODO Outsource to tests
                GLOG() << "[A]" << l << "," << m << " : " << out[l*l+l+m] << std::endl;
            #endif
        }
    }
    return out;
}

std::vector<double> calculate_Blm(int L) {
    // B_{lm} = A_{lm} / (2l-1)!!
    std::vector<double> out;
    out.resize((L+1)*(L+1), 0.0);
    for (int l = 0; l <= L; ++l) {
        for (int m = -l; m <= l; ++m) {
            out[l*l+l+m] = sqrt(factorial(l+m)*factorial(l-m))/factorial2(2*l-1);
            #ifdef DBG // TODO Outsource to tests
                GLOG() << "[B]" << l << "," << m << " : " << out[l*l+l+m] << std::endl;
            #endif
        }
    }
    return out;
}

void calculate_Fl(double r, double a, int L, std::vector<double> &fl) {
    // See Elking et al.: "Gaussian Multipole Model", JCTC (2010), Eq. 15a++
    // (which does not however mention the recursive form used here)
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

void Tlmlm::computeTlmlm(vec d12, double r12, double a12, int L1, int L2, coeff_t &T) {
    // TODO This can be sped up by using Tl1m1l2m2 = (-1)^(l1+l2) Tl2m2l1m1
    // TODO Properties to test:
    //      Tl1l2 ~ (l-1)! * 1/r^{l1+l2+1} for large r
    //      Tl1l2 = (-1)^{l1+l2} Tl2l1 and T(-r) = (-1)^{l1+l2} T(r)
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

    /*
    for (int l1 = 0; l1 <= Lg; ++l1) {
        std::complex<double> sum_l = 0.0;
        for (int m1 = -l1; m1 <= l1; ++m1) {
            int lm1 = l1*l1+l1+m1;
            GLOG() << "[Rlm] " << " m1 " << l1 << " : " << Rlm[lm1] << std::endl;
            sum_l += Rlm[lm1]*std::conj(Rlm[lm1]);
        }
        GLOG() << "[Rlm] => " << r12 << " l " << l1 << " : " << sum_l << std::endl;
    }
    for (int l1 = 0; l1 <= L1; ++l1) {
    for (int l2 = 0; l2 <= L2; ++l2) {
        std::complex<double> sum_l1_l2 = 0.0;
        for (int m1 = -l1; m1 <= l1; ++m1) {
            int lm1 = l1*l1+l1+m1;
        for (int m2 = -l2; m2 <= l2; ++m2) {
            int lm2 = l2*l2+l2+m2;
            sum_l1_l2 += T(lm1,lm2)*std::conj(T(lm1,lm2));
        }}
        GLOG() << "[Tlmlm] => " << l1 << ":" << l2 << " sum " << sum_l1_l2 << std::endl;
    }}
    */

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

            //tlm1lm2 +=
            //      pow(-1, l2+m)*1./(Alm[lm]*Blm[lm])
            //    * std::conj( Rlm[ (l2-l)*(l2-l) + (l2-l) + m2+m ] ) / Alm[(l2-l)*(l2-l) + (l2-l) + m2+m ]
            //    * std::conj( Rlm[ (l1-l)*(l1-l) + (l1-l) + m1-m ] ) / Alm[(l1-l)*(l1-l) + (l1-l) + m1-m ]
            //    * Fl[ l1+l2-l ];

        }} // l m

        tlm1lm2 *=
              Alm[lm1]*Alm[lm1] * Alm[lm2]*Alm[lm2]
           / (factorial2(2*l1-1)*factorial2(2*l2-1));
        tlm1lm2 /=               // HACK Is that the fix? Yes. TODO Adjust scaling
            Blm[lm1]*Blm[lm2];   // HACK Is that the fix? Yes. TODO Adjust scaling
        T(lm1,lm2) = tlm1lm2;
    }}}} // l1 m1 l2 m2
    return;
}

FTSpectrum::FTSpectrum(Structure &structure, Options &options)
    : _structure(&structure), _options(&options) {
    #ifdef DBG
        GLOG() << "FTSPECTRUM: DEBUG MODE ENABLED" << std::endl;
    #endif
    _cutoff = CutoffFunctionOutlet().create(_options->get<std::string>("radialcutoff.type"));
	_cutoff->configure(*_options);
    _L = _options->get<int>("fieldtensor.L");
    _K = _options->get<int>("fieldtensor.K");
    return;
}

FTSpectrum::~FTSpectrum() {
    for (auto it = _atomic_array.begin(); it != _atomic_array.end(); ++it) {
        delete *it;
    }
    return;
}

void InteractUpdateSourceTarget(
        AtomicSpectrumFT *a, // <- source particle
        AtomicSpectrumFT *b, // <- target particle
        Tlmlm::coeff_t &T_ab, // <- field tensors
        double sigma1,      // <- atomic width source
        double sigma2,      // <- atomic width target
        double w12,         // <- pair weight
        double w2,          // <- target weight
        int k,              // <- Body-order cutoff
        int L_in,           // <- Incoming moments: cutoff
        int L_out,          // <- Outgoing moments: cutoff
        int Lambda_in,      // <- Trace memory length in
        int Lambda_out,     // <- Trace memory length out
        int LM_in,          // <- (no longer used)
        int LM_out,         // <- Allocation size out
        bool apply_parity) { // <- apply parity to field tensors

    #ifdef DBG
        GLOG() << "[update] " << "w " << w12 << " " << w2 << std::endl;
        GLOG() << "[update] " << "k " << k << " L-in " << L_in << " L-out " << L_out << std::endl;
        GLOG() << "[update] " << "Lambda-in " << Lambda_in << " Lambda-out " << Lambda_out << std::endl;
        GLOG() << "[update] " << "LM-in " << LM_in << " LM-out " << LM_out << std::endl;
        GLOG() << "[update] " << "Parity " << apply_parity << std::endl;
        for (int l1=0; l1 <= L_out; ++l1) {
        for (int m1=-l1; m1 <= l1; ++m1) {
            int lm1 = l1*l1+l1+m1;
            for (int l2=0; l2 <= L_out; ++l2) {
            for (int m2=-l2; m2 <= l2; ++m2) {
                int lm2 = l2*l2+l2+m2;
                GLOG() << l1 << "," << m1
                    << " " << l2 << "," << m2
                    << " " << T_ab(lm1,lm2)
                    << std::endl;
            }}
        }}
    #endif

    // Retrieve field maps for iteration k-1
    AtomicSpectrumFT::field_map_t &fmap_a = a->getFieldMap(k-1);
    AtomicSpectrumFT::field_map_t &fmap_b = b->getFieldMap(k-1);
    // TARGET: MOMENTS ON 'b'
    // Loop over field channels/types: a
    for (auto it_a = fmap_a.begin(); it_a != fmap_a.end(); ++it_a) {
        std::string type = it_a->first;
        AtomicSpectrumFT::field_t &f_a = *(it_a->second);
        AtomicSpectrumFT::field_t &f_b = *(b->getCreateField(k, type, Lambda_out, LM_out));
        #ifdef DBG
            GLOG() << "[increment]   a: source: type: "
                << type << " size: " << f_a.size1()
                << " x " << f_a.size2()
                << std::endl;
            GLOG() << "[increment]   b: target: size: "
                << f_b.size1() << " x " << f_b.size2() << std::endl;
        #endif
        // lm-out
        for (int l1=0; l1<=L_out; ++l1) {
            // Polarizability
            //double alpha = (2*l1+1)*pow(sigma2, 2*l1+1); // TODO HACK: 2*l1+1
            //double alpha = (2*l1+1)*pow(sigma2, l1);
            //double alpha = (l1+0.1)*pow(sigma2, l1);
            double alpha = b->_alpha[l1];
            for (int m1=-l1; m1<=l1; ++m1) {
                int lm_out = l1*l1+l1+m1;
                // lambda-in
                for (int lambda_in=0; lambda_in<Lambda_in; ++lambda_in) {
                    // lm-in
                    for (int l2=0; l2<=L_in; ++l2) {
                        double parity = (apply_parity) ? pow(-1,l1+l2) : 1;
                        int lambda_out = lambda_in*Lambda_in+l2;
                        AtomicSpectrumFT::cmplx_t q_lambda_lm_out = 0.0;
                        for (int m2=-l2; m2<=l2; ++m2) {
                            #ifdef DBG
                                GLOG() << "l_out " << l1
                                    << " m_out " << m1
                                    << " lambda_in " << lambda_in
                                    << " l_in " << l2
                                    << " m_in " << m2
                                    << std::endl;
                            #endif
                            int lm_in = l2*l2+l2+m2;
                            //q_lambda_lm_out += T_ab(lm_out, lm_in)*f_a(lambda_in, lm_in); // BACK-UP
                            //q_lambda_lm_out += pow(sigma1, l1+0.5)*pow(sigma2, l2+0.5)*T_ab(lm_out, lm_in)*f_a(lambda_in, lm_in); // TODO HACK
                            q_lambda_lm_out += pow(sigma1, -l1)*pow(sigma2, l2)*T_ab(lm_out, lm_in)*f_a(lambda_in, lm_in); // TODO HACK
                        }
                        // Apply weight and parity
                        q_lambda_lm_out *= alpha*w12*w2*parity;
                        // Add to fields on b
                        #ifdef DBG
                            GLOG() << "    => (" << lambda_out << "," << lm_out << ")"
                                << " : " << q_lambda_lm_out
                                << std::endl;
                        #endif
                        f_b(lambda_out, lm_out) += std::conj(q_lambda_lm_out); // FIXED Added conj
                    } // End loop over l_in
                } // End loop over lambda_in
            } // End loop over m_out
        } // End loop over l_out
    } // End loop over types on a
    return;
}

void FTSpectrum::computeFieldTensors(std::map<int, std::map<int, Tlmlm::coeff_t>> &i1_i2_T12) {
    GLOG() << "Computing field tensors ..." << std::endl;
    int K = _K;
    int L = _L;
    Tlmlm T12(L);
    for (auto it1 = beginAtomic(); it1 != endAtomic(); ++it1) {
        // Particle 1
        atomic_t *a = *it1;
        int id1 = a->getCenter()->getId();
        int s1 = a->getTypeIdx();
        vec r1 = a->getCenter()->getPos();
        double sigma1 = a->getCenter()->getSigma();
        // Initialise map
        i1_i2_T12[id1] = std::map<int, Tlmlm::coeff_t>();
        for (auto it2 = it1; it2 != endAtomic(); ++it2) {
            // Particle 2
            atomic_t *b = *it2;
            int id2 = b->getCenter()->getId();
            int s2 = b->getTypeIdx();
            vec r2 = b->getCenter()->getPos();
            double sigma2 = b->getCenter()->getSigma();
            // Find connection, apply weight function
            vec dr12 = _structure->connect(r1, r2);
            double r12 = soap::linalg::abs(dr12);
            vec d12 = dr12/r12;
            if (! _cutoff->isWithinCutoff(r12)) continue;
            double w12 = _cutoff->calculateWeight(r12);
            double a12 = 1./sqrt(2*(sigma1*sigma1 + sigma2*sigma2));
            #ifdef DBG
                GLOG() << "[field-tensor] " << a->getCenter()->getId()
                    << ":" << b->getCenter()->getId()
                    << " R=" << r12
                    << " w=" << w12
                    << " a=" << a12
                    << std::endl;
            #endif
            // Interact
            T12.computeTlmlm(d12, r12, a12, L, L, i1_i2_T12[id1][id2]);
        }
    }
    return;
}

void FTSpectrum::createAtomic() {
    for (auto it = _atomic_array.begin(); it != _atomic_array.end(); ++it) {
        delete *it;
    }
    _atomic_array.clear();
    for (auto pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
        atomic_t *new_atomic = new atomic_t(*pit, _K, _L, _options);
        _atomic_array.push_back(new_atomic);
    }
}

void FTSpectrum::energySCF(int k, std::map<int, std::map<int, Tlmlm::coeff_t> > &i1_i2_T12) {
    std::complex<double> energy_total;
    for (auto it1 = beginAtomic(); it1 != endAtomic(); ++it1) {
        // Particle 1
        atomic_t *a = *it1;
        int id1 = a->getCenter()->getId();
        std::string s1 = a->getType();
        vec r1 = a->getCenter()->getPos();
        double sigma1 = a->getCenter()->getSigma();
        double w1 = a->getCenter()->getWeight();
        for (auto it2 = it1; it2 != endAtomic(); ++it2) {
            // Particle 2
            atomic_t *b = *it2;
            int id2 = b->getCenter()->getId();
            std::string s2 = b->getType();
            vec r2 = b->getCenter()->getPos();
            double sigma2 = b->getCenter()->getSigma();
            double w2 = b->getCenter()->getWeight();
            // Find connection, apply weight function
            vec dr12 = _structure->connect(r1, r2);
            double r12 = soap::linalg::abs(dr12);
            vec d12 = dr12/r12;
            if (! _cutoff->isWithinCutoff(r12)) continue;
            double w12 = _cutoff->calculateWeight(r12);
            double a12 = 1./sqrt(2*(sigma1*sigma1 + sigma2*sigma2));
            // Look up interaction tensors for pair (a,b)
            auto T_ab = i1_i2_T12[id1][id2];
            // Self-interaction a <> a
            if (a == b) {
                ;
            }
            // Cross-interaction a <> b
            else {
                std::complex<double> energy = 0.0;
                for (int l1=0; l1<=_L; ++l1) {
                    for (int m1=-l1; m1<=l1; ++m1) {
                        int lm1 = l1*l1+l1+m1;
                        for (int l2=0; l2<=_L; ++l2) {
                            for (int m2=-l2; m2<=l2; ++m2) {
                                int lm2 = l2*l2+l2+m2;
                                energy += a->_M(k, lm1)*T_ab(lm1, lm2)*b->_M(k, lm2);
                            }
                        }
                    }
                }
                energy_total += energy;
                GLOG() << "[energy] " << a->getCenter()->getId()
                    << ":" << b->getCenter()->getId() << " " << energy
                    << std::endl;
            }
        } // Particle b
    } // Particle a
    GLOG() << "[energy, total] " << "(k=" << k << ")" << energy_total << std::endl;
    return;
}

void FTSpectrum::polarize() {
    this->createAtomic();
    std::map<int, std::map<int, Tlmlm::coeff_t>> i1_i2_T12;
    this->computeFieldTensors(i1_i2_T12);

    // Initialise moments
    for (auto it = beginAtomic(); it != endAtomic(); ++it) {
        atomic_t *a = *it;
        a->_M = AtomicSpectrumFT::field_zero_t(_K+1,(_L+1)*(_L+1));
        a->_F = AtomicSpectrumFT::field_zero_t(_K+1,(_L+1)*(_L+1));
        a->_M(0,0) = 1.0;
    }
    this->energySCF(0, i1_i2_T12);

    for (int k = 1; k <= _K; ++k) {
        // UPDATE FIELDS
        for (auto it1 = beginAtomic(); it1 != endAtomic(); ++it1) {
            // Particle 1
            atomic_t *a = *it1;
            int id1 = a->getCenter()->getId();
            std::string s1 = a->getType();
            vec r1 = a->getCenter()->getPos();
            double sigma1 = a->getCenter()->getSigma();
            double w1 = a->getCenter()->getWeight();
            for (auto it2 = it1; it2 != endAtomic(); ++it2) {
                // Particle 2
                atomic_t *b = *it2;
                int id2 = b->getCenter()->getId();
                std::string s2 = b->getType();
                vec r2 = b->getCenter()->getPos();
                double sigma2 = b->getCenter()->getSigma();
                double w2 = b->getCenter()->getWeight();
                // Find connection, apply weight function
                vec dr12 = _structure->connect(r1, r2);
                double r12 = soap::linalg::abs(dr12);
                vec d12 = dr12/r12;
                if (! _cutoff->isWithinCutoff(r12)) continue;
                double w12 = _cutoff->calculateWeight(r12);
                double a12 = 1./sqrt(2*(sigma1*sigma1 + sigma2*sigma2));
                GLOG() << "[interact] " << a->getCenter()->getId()
                    << ":" << b->getCenter()->getId()
                    << " R=" << r12
                    << " w=" << w12
                    << " a=" << a12
                    << std::endl;
                // Look up interaction tensors for pair (a,b)
                auto T_ab = i1_i2_T12[id1][id2];
                for (int l1=0; l1 <= _L; ++l1) {
                    for (int m1=-l1; m1 <= l1; ++m1) {
                        int lm1 = l1*l1+l1+m1;
                        for (int l2=0; l2 <= _L; ++l2) {
                            for (int m2=-l2; m2 <= l2; ++m2) {
                                int lm2 = l2*l2+l2+m2;
                                #ifdef DBG
                                    GLOG() << "[FT] " << id1<<":"<<id2
                                        << " " << l1 << "," << m1
                                        << " " << l2 << "," << m2
                                        << " " << T_ab(lm1,lm2)
                                        << std::endl;
                                #endif
                            }
                        }
                    }
                }
                // Self-interaction a <> a
                if (a == b) {
                    ;
                }
                // Cross-interaction a <> b
                else {
                    for (int l1=0; l1<=_L; ++l1) {
                        for (int m1=-l1; m1<=l1; ++m1) {
                            int lm1 = l1*l1+l1+m1;
                            for (int l2=0; l2<=_L; ++l2) {
                                double parity = pow(-1,l1+l2);
                                for (int m2=-l2; m2<=l2; ++m2) {
                                    int lm2 = l2*l2+l2+m2;
                                    a->_F(k, lm1) += T_ab(lm1, lm2)*b->_M(k-1, lm2);
                                    b->_F(k, lm1) += T_ab(lm1, lm2)*a->_M(k-1, lm2)*parity;
                                    #ifdef DBG
                                        GLOG() << "@ " << id1 << " ~" << id2
                                            << " " << l1 << m1 << "|" << l2 << m2 << " : "
                                            << T_ab(lm1, lm2)*b->_M(k-1, lm2)
                                            << std::endl;
                                        GLOG() << "@ " << id2 << " ~" << id1
                                            << " " << l1 << m1 << "|" << l2 << m2 << " : "
                                            << T_ab(lm1, lm2)*a->_M(k-1, lm2)*parity
                                            << std::endl;
                                    #endif
                                }
                            }
                        }
                    }
                }
            } // Particle b
        } // Particle a
        // INDUCE
        for (auto it1 = beginAtomic(); it1 != endAtomic(); ++it1) {
            atomic_t *a = *it1;
            double sigma1 = a->getCenter()->getSigma();
            for (int l1=0; l1<=_L; ++l1) {
                std::complex<double> sum_l = 0.0;
                double alpha = pow(sigma1, 2*l1+1);
                for (int m1=-l1; m1<=l1; ++m1) {
                    int lm1 = l1*l1+l1+m1;
                    a->_M(k, lm1) = std::conj(alpha*a->_F(k, lm1)); // FIXED added conj
                    sum_l += a->_F(k, lm1)*std::conj(a->_F(k, lm1));
                    #ifdef DBG
                        GLOG() << "[induced] " << "l " << l1 << " m " << m1
                            << " lm " << lm1 <<  " : "  << a->_F(k, lm1)
                            << std::endl;
                    #endif
                }
                GLOG() << "[induced] @ " << a->getCenter()->getId()
                    << " l " << l1 << " : " << sum_l
                    << std::endl;
            }
        }
        // ENERGY
        this->energySCF(k, i1_i2_T12);
    } // Iterations k
    return;
}

void FTSpectrum::compute() {
    GLOG() << "Computing FTSpectrum ..." << std::endl;

    // CREATE ATOMIC SPECTRA
    this->createAtomic();

    // COMPUTE INTERACTION TENSORS
    std::map<int, std::map<int, Tlmlm::coeff_t>> i1_i2_T12;
    this->computeFieldTensors(i1_i2_T12);

    // INITIALISE FIELD MOMENTS (k=0)
    int K = _K;
    int L = _L;
    int k = 0;
    int L_k_in = 0;
    int L_k_out = 0;
    int Lambda_flat_k_in = 1;
    int Lambda_flat_k_out = 1; // || l'l''... ||
    GLOG() << "Initialise fields (k=0) ..." << std::endl;
    for (auto it = beginAtomic(); it != endAtomic(); ++it) {
        atomic_t *a = *it;
        std::string s = a->getType();
        AtomicSpectrumFT::field_t flm
            = AtomicSpectrumFT::field_zero_t(Lambda_flat_k_out, (L_k_in+1)*(L_k_in+1));
        #ifdef DBG
            GLOG() << "[init] Center:" << a->getCenter()->getId() << std::endl;
            GLOG() << "[init]   k = " << k << std::endl;
            GLOG() << "[init]   s = " << s << std::endl;
            GLOG() << "[init]   F : " << flm.size1() << " x " << flm.size2() << std::endl;
        #endif
        flm(0,0) = 1.0;
        a->addField(k, s, flm);
    }

    // PROPAGATE FIELD MOMENTS (k>0)
    for (int k = 1; k <= K; ++k) {

        GLOG() << "Starting iteration: k = " << k << std::endl;
        // Update out-going index ranges
        L_k_out = L;
        Lambda_flat_k_out = Lambda_flat_k_out*(L_k_in+1);
        int LM_in = (L_k_in+1)*(L_k_in+1);
        int LM_out = (L_k_out+1)*(L_k_out+1);
        #ifdef DBG
            GLOG() << "[iter] L-in " << L_k_in
                << " L-out " << L_k_out
                << " Lambda-in " << Lambda_flat_k_in
                << " Lambda-out " << Lambda_flat_k_out
                << std::endl;
            GLOG() << "[iter] Rank: " << L
                << " (LM-in = " << LM_in << ")"
                << " (LM-out = " << LM_out << ")"
                << std::endl;
        #endif
        for (auto it1 = beginAtomic(); it1 != endAtomic(); ++it1) {
            // Particle 1
            atomic_t *a = *it1;
            int id1 = a->getCenter()->getId();
            std::string s1 = a->getType();
            vec r1 = a->getCenter()->getPos();
            double sigma1 = a->getCenter()->getSigma();
            double w1 = a->getCenter()->getWeight();
            for (auto it2 = it1; it2 != endAtomic(); ++it2) {
                // Particle 2
                atomic_t *b = *it2;
                int id2 = b->getCenter()->getId();
                std::string s2 = b->getType();
                vec r2 = b->getCenter()->getPos();
                double sigma2 = b->getCenter()->getSigma();
                double w2 = b->getCenter()->getWeight();
                // Find connection, apply weight function
                vec dr12 = _structure->connect(r1, r2);
                double r12 = soap::linalg::abs(dr12);
                vec d12 = dr12/r12;
                if (! _cutoff->isWithinCutoff(r12)) continue;
                double w12 = _cutoff->calculateWeight(r12);
                double a12 = 1./sqrt(2*(sigma1*sigma1 + sigma2*sigma2));
                if (GLOG.verbose())
                    GLOG() << "[interact] " << a->getCenter()->getId()
                        << ":" << b->getCenter()->getId()
                        << " R=" << r12
                        << " w=" << w12
                        << " a=" << a12
                        << std::endl;
                // Look up interaction tensors for pair (a,b)
                auto T_ab = i1_i2_T12[id1][id2];
                // Self-interaction a <> a
                if (a == b) {
                    w12 *= _cutoff->getCenterWeight();
                    // Source 'a', target 'a'
                    InteractUpdateSourceTarget(a, b, T_ab,
                        sigma1, sigma2,
                        w12, w2,
                        k, L_k_in, L_k_out,
                        Lambda_flat_k_in, Lambda_flat_k_out,
                        LM_in, LM_out,
                        false);
                }
                // Cross-interaction a <> b
                else {
                    // Source 'a', target 'b'
                    InteractUpdateSourceTarget(a, b, T_ab,
                        sigma1, sigma2,
                        w12, w2,
                        k, L_k_in, L_k_out,
                        Lambda_flat_k_in, Lambda_flat_k_out,
                        LM_in, LM_out,
                        true);
                    // Source 'b', target 'a'
                    InteractUpdateSourceTarget(b, a, T_ab,
                        sigma1, sigma2,
                        w12, w1,
                        k, L_k_in, L_k_out,
                        Lambda_flat_k_in, Lambda_flat_k_out,
                        LM_in, LM_out,
                        false);
                }
            }
        }
        // Update dimension of flat indices
        Lambda_flat_k_in = Lambda_flat_k_in*(L_k_in+1);
        L_k_in = L;
    }

    if (_options->get<std::string>("fieldtensor.contract") == "normal") {
        for (auto it = beginAtomic(); it != endAtomic(); ++it) {
            (*it)->contract();
        }
    }
    else if (_options->get<std::string>("fieldtensor.contract") == "deep") {
        for (auto it = beginAtomic(); it != endAtomic(); ++it) {
            (*it)->contractDeep();
        }
    }
    else {
        assert(false && "fieldtensor.contract has invalid value (should be 'normal' or 'deep')");
    }

    /*
    //Tlmlm T12(L);
    vec d12(0.,0.,1.);
    double a12 = 0.5;
    int L1 = 2;
    int L2 = 2;
    for (double r12 = 0.; r12 <= 10.; r12 += 0.05) {
        Tlmlm::coeff_t T_mat;
        T12.computeTlmlm(d12, r12, a12, L1, L2, T_mat);

        int l1 = 1;
        int m1 = 0;
        int l2 = 1;
        int m2 = 0;

        int lm1 = l1*l1+l1+m1;
        int lm2 = l2*l2+l2+m2;
        GLOG() << r12 << " " << T_mat(lm1,lm2).real() << " " << T_mat(lm2,lm1).real()
        << " " << T_mat(lm1,lm2).imag() << " " << T_mat(lm2,lm1).imag() << std::endl;
    }
    */

    /*GLOG() << "Factorial2 ..." << std::endl;
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
    */

    return;
}

void FTSpectrum::registerPython() {
    using namespace boost::python;
    class_<FTSpectrum>("FTSpectrum", init<Structure &, Options &>())
        .def("__iter__", range<return_value_policy<reference_existing_object> >(
            &FTSpectrum::beginAtomic, &FTSpectrum::endAtomic))
        .def("polarize", &FTSpectrum::polarize)
        .def("compute", &FTSpectrum::compute);
    return;
}

} // Namespace
