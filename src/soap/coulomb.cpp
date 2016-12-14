#include "soap/coulomb.hpp"
#include "soap/linalg/numpy.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

// =====================================
// Spectrum::HierarchicalCoulomb::Atomic
// =====================================

const std::string AtomicSpectrumHC::_numpy_t = "float64";

AtomicSpectrumHC::AtomicSpectrumHC(Particle *center, int S)
    : _center(center), _S(S) {
    this->_Q0 = coeff_zero_t(S,1);
    this->_Q1 = coeff_zero_t(S,S);
    this->_Q2 = coeff_zero_t(S,S*S);
    this->_Q3 = coeff_zero_t(S,S*S*S);

    this->_s = center->getTypeId()-1;
    assert(_s >= 0 && "Type-IDs should start from 1");
}

AtomicSpectrumHC::~AtomicSpectrumHC() {
    _Q0.clear();
    _Q1.clear();
    _Q2.clear();
    _Q3.clear();
}

void AtomicSpectrumHC::setCoefficientsNumpy_k1(boost::python::object &np_array) {
    throw soap::base::APIError("<AtomicSpectrumHC::setCoefficientsNumpy> Not implemented.");
}

void AtomicSpectrumHC::setCoefficientsNumpy_k2(boost::python::object &np_array) {
    throw soap::base::APIError("<AtomicSpectrumHC::setCoefficientsNumpy> Not implemented.");
}

void AtomicSpectrumHC::setCoefficientsNumpy_k3(boost::python::object &np_array) {
    throw soap::base::APIError("<AtomicSpectrumHC::setCoefficientsNumpy> Not implemented.");
}

void AtomicSpectrumHC::setCoefficientsNumpy_k4(boost::python::object &np_array) {
    throw soap::base::APIError("<AtomicSpectrumHC::setCoefficientsNumpy> Not implemented.");
}

boost::python::object AtomicSpectrumHC::getCoefficientsNumpy_k1() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_Q0);
}

boost::python::object AtomicSpectrumHC::getCoefficientsNumpy_k2() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_Q1);
}

boost::python::object AtomicSpectrumHC::getCoefficientsNumpy_k3() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_Q2);
}

boost::python::object AtomicSpectrumHC::getCoefficientsNumpy_k4() {
    soap::linalg::numpy_converter npc(_numpy_t.c_str());
    return npc.ublas_to_numpy< dtype_t >(_Q3);
}

void AtomicSpectrumHC::registerPython() {
    using namespace boost::python;
    class_<AtomicSpectrumHC, AtomicSpectrumHC*>("AtomicSpectrumHC", init<Particle*, int>())
        .add_property("array_k1", &AtomicSpectrumHC::getCoefficientsNumpy_k1, &AtomicSpectrumHC::setCoefficientsNumpy_k1)
        .add_property("array_k2", &AtomicSpectrumHC::getCoefficientsNumpy_k2, &AtomicSpectrumHC::setCoefficientsNumpy_k2)
        .add_property("array_k3", &AtomicSpectrumHC::getCoefficientsNumpy_k3, &AtomicSpectrumHC::setCoefficientsNumpy_k3)
        .add_property("array_k4", &AtomicSpectrumHC::getCoefficientsNumpy_k4, &AtomicSpectrumHC::setCoefficientsNumpy_k4)
        .def("getArray_k1", &AtomicSpectrumHC::getCoefficientsNumpy_k1)
        .def("getArray_k2", &AtomicSpectrumHC::getCoefficientsNumpy_k2)
        .def("getArray_k3", &AtomicSpectrumHC::getCoefficientsNumpy_k3)
        .def("getArray_k4", &AtomicSpectrumHC::getCoefficientsNumpy_k4);
}

// =============================
// Spectrum::HierarchicalCoulomb
// =============================

HierarchicalCoulomb::HierarchicalCoulomb(Structure &structure, Options &options) {
    GLOG() << "Creating HierarchicalCoulomb ..." << std::endl;
    _structure = &structure;
    _options = &options;
    _cutoff = CutoffFunctionOutlet().create(_options->get<std::string>("radialcutoff.type"));
	_cutoff->configure(*_options);
    _S = 5; // TODO
    return;
}

HierarchicalCoulomb::~HierarchicalCoulomb() {
    for (auto it = _atomic_array.begin(); it != _atomic_array.end(); ++it) {
        delete *it;
    }
    _atomic_array.clear();
    return;
}

void HierarchicalCoulomb::compute() {
    GLOG() << "Computing HierarchicalCoulomb ..." << std::endl;

    // CLEAN EXISTING
    for (auto it = _atomic_array.begin(); it != _atomic_array.end(); ++it) {
        delete *it;
    }
    _atomic_array.clear();
    // CREATE ATOMIC SPECTRA
    for (auto pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
        atomic_t *new_atomic = new atomic_t(*pit, _S);
        _atomic_array.push_back(new_atomic);
    }

    double R0 = _options->get<double>("hierarchicalcoulomb.r0");
    double gamma = _options->get<double>("hierarchicalcoulomb.gamma");
    bool norm = _options->get<bool>("hierarchicalcoulomb.norm");

    // ZEROETH-ORDER
    GLOG() << "k = 1-body ..." << std::endl;
    for (auto it = beginAtomic(); it != endAtomic(); ++it) {
        atomic_t *a = *it;
        int sa = a->getTypeIdx();
        double wa = a->getCenter()->getWeight();
        GLOG() << "    " << a->getCenter()->getId() << std::endl;
        a->_Q0(sa, 0) = wa/pow(R0, gamma);
    }
    if (norm) {
        for (auto it = beginAtomic(); it != endAtomic(); ++it) {
            atomic_t *a = *it;
            a->_Q0 = a->_Q0/ub::norm_frobenius(a->_Q0);
        }
    }

    // FIRST-ORDER
    GLOG() << "k = 2-body ..." << std::endl;
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
            if (! _cutoff->isWithinCutoff(R_ab)) continue;
            double w_ab = _cutoff->calculateWeight(R_ab);
            GLOG() << "    " << a->getCenter()->getId() << ":" << b->getCenter()->getId() << " R=" << R_ab << " w=" << w_ab << std::endl;
            // Interact
            double inter_ab = w_ab/pow(R_ab+R0, gamma);
            a->_Q1(sa, sb) += (
                a->_Q0(sa,0)*b->_Q0(sb,0)
            )*inter_ab;
        }
    }
    if (norm) {
        for (auto it = beginAtomic(); it != endAtomic(); ++it) {
            atomic_t *a = *it;
            a->_Q1 = a->_Q1/ub::norm_frobenius(a->_Q1);
        }
    }
    // SECOND-ORDER
    GLOG() << "k = 3-body ..." << std::endl;
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
            if (! _cutoff->isWithinCutoff(R_ab)) continue;
            double w_ab = _cutoff->calculateWeight(R_ab);
            GLOG() << "    " << a->getCenter()->getId() << ":" << b->getCenter()->getId() << " R=" << R_ab << " w=" << w_ab << std::endl;
            // Interact
            double inter_ab = w_ab/pow(R_ab+R0, gamma);
            for (int sb = 0; sb < _S; ++sb) {
                for (int sc = 0; sc < _S; ++sc) {
                    a->_Q2(sa, sb*_S+sc) += (
                        a->_Q1(sa,sb)  *b->_Q0(sc,0) + // runs over sb
                        a->_Q0(sa,0)   *b->_Q1(sb,sc)  // runs over sc
                    )*inter_ab;
                }
            }
        }
    }
    if (norm) {
        for (auto it = beginAtomic(); it != endAtomic(); ++it) {
            atomic_t *a = *it;
            a->_Q2 = a->_Q2/ub::norm_frobenius(a->_Q2);
        }
    }

    // THIRD-ORDER
    GLOG() << "k = 4-body ..." << std::endl;
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
            if (! _cutoff->isWithinCutoff(R_ab)) continue;
            double w_ab = _cutoff->calculateWeight(R_ab);
            GLOG() << "    " << a->getCenter()->getId() << ":" << b->getCenter()->getId() << " R=" << R_ab << " w=" << w_ab << std::endl;
            // Interact
            double inter_ab = w_ab/pow(R_ab+R0, gamma);
            for (int sb = 0; sb < _S; ++sb) {
                for (int sc = 0; sc < _S; ++sc) {
                    for (int sd = 0; sd < _S; ++sd) {
                        a->_Q3(sa, sb*_S*_S+sc*_S+sd) += (
                            a->_Q0(sa,0)         *b->_Q2(sb, sc*_S+sd) + // runs over sc, sd
                            a->_Q1(sa,sb)        *b->_Q1(sc, sd) +       // runs over sb, sd
                            a->_Q2(sa,sb*_S+sc)  *b->_Q0(sd,0)           // runs over sb, sc
                        )*inter_ab;
                    }
                }
            }
        }
    }

    if (norm) {
        for (auto it = beginAtomic(); it != endAtomic(); ++it) {
            atomic_t *a = *it;
            a->_Q3 = a->_Q3/ub::norm_frobenius(a->_Q3);
        }
    }

    return;
}

void HierarchicalCoulomb::registerPython() {
    using namespace boost::python;
    class_<HierarchicalCoulomb>("HierarchicalCoulomb", init<Structure &, Options &>())
        .def("__iter__", range<return_value_policy<reference_existing_object> >(&HierarchicalCoulomb::beginAtomic, &HierarchicalCoulomb::endAtomic))
        .def("compute", &HierarchicalCoulomb::compute);
    return;
}

}
