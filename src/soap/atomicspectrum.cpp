#include <fstream>
#include <boost/format.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "soap/atomicspectrum.hpp"

namespace soap {

AtomicSpectrum::AtomicSpectrum(Particle *center, Basis *basis) {
    this->null();
    _center = center;
    _center_id = center->getId();
    _center_pos = center->getPos();
    _center_type = center->getType();
    _basis = basis;
    _qnlm_generic = new BasisExpansion(_basis);
}

AtomicSpectrum::AtomicSpectrum(Basis *basis) {
    this->null();
    _basis = basis;
    _qnlm_generic = new BasisExpansion(_basis);
}

AtomicSpectrum::~AtomicSpectrum() {
    // CLEAN QNLM'S
    // ... Summed density expansions
    for (auto it = _map_qnlm.begin(); it != _map_qnlm.end(); ++it) delete it->second;
    _map_qnlm.clear();
    // ... Generic density
    if (_qnlm_generic) {
        delete _qnlm_generic;
        _qnlm_generic = NULL;
    }
    // CLEAN XNKL'S
    // ... Scalar spectra
    for (auto it = _map_xnkl.begin(); it != _map_xnkl.end(); ++it) delete it->second;
    _map_xnkl.clear();

    // ... Generic power spectra
    if (_xnkl_generic_coherent) {
        delete _xnkl_generic_coherent;
        _xnkl_generic_coherent = NULL;
    }
    if (_xnkl_generic_incoherent) {
        delete _xnkl_generic_incoherent;
        _xnkl_generic_incoherent = NULL;
    }
    // CLEAN PID-RESOLVED GRADIENTS
    this->prunePidData();
}

void AtomicSpectrum::null() {
    _center = NULL;
    _center_id = -1;
    _center_pos = vec(0,0,0);
    _center_type = "?";
    _basis = NULL;
    _qnlm_generic = NULL;
    _xnkl_generic_coherent = NULL;
    _xnkl_generic_incoherent = NULL;
}

void AtomicSpectrum::prunePidData() {
    // CLEAN PID-RESOLVED GRADIENTS
    // ... Neighbour density expansions with gradients
    for (auto it = _map_pid_qnlm.begin(); it != _map_pid_qnlm.end(); ++it) delete it->second.second;
    _map_pid_qnlm.clear();
    // ... Xnkl gradients
    for (auto it = _map_pid_xnkl.begin(); it != _map_pid_xnkl.end(); ++it) {
        for (auto jt = (*it).second.begin(); jt != (*it).second.end(); ++jt) {
            delete (*jt).second;
        }
        (*it).second.clear();
    }
    _map_pid_xnkl.clear();
    // ... Gradients (generic-coherent)
    for (auto it = _map_pid_xnkl_gc.begin(); it != _map_pid_xnkl_gc.end(); ++it) {
        delete it->second;
    }
}

void AtomicSpectrum::invert(map_xnkl_t &map_xnkl, xnkl_t *xnkl_generic_coherent, std::string type1, std::string type2) {

    int N = _basis->getRadBasis()->N();
    int L = _basis->getAngBasis()->L();

    assert(xnkl_generic_coherent->getBasis() == _basis &&
        "Trying to invert spectrum linked against foreign basis.");

    PowerExpansion::coeff_t &xnkl = xnkl_generic_coherent->getCoefficients();
    BasisExpansion::coeff_t &qnlm = _qnlm_generic->getCoefficients();

    type_pair_t types(type1, type2);
    if (type1 == "g" && type2 == "c") xnkl = xnkl_generic_coherent->getCoefficients();
    else if (type1 == "g" && type2 == "i") throw soap::base::NotImplemented("::invert g/i");
    else xnkl = map_xnkl[types]->getCoefficients();

    // ZERO QNLM TO BE SAFE
    for (int n= 0; n < N; ++n) {
        for (int l = 0; l <= L; ++l) {
            for (int m = -l; m <= l; ++m) {
                qnlm(n, l*l+l+m) = std::complex<double>(0.,0.);
            }
        }
    }
    // Q000 from X000, then Qk00 = X0k0/Q000
    typedef std::complex<double> cmplx;
    int n = 0;
    int k = 0;
    int l = 0;
    int m = 0;
//    qnlm(n, l*l+l+m) = cmplx(sqrt(xnkl(n*N+k, l).real()), 0.);
//    for (k = 1; k < N; ++k) {
//        qnlm(k, l*l+l+m) = cmplx(xnkl(n*N+k, l).real()/qnlm(0, l*l+l+m).real(), 0.);
//    }

//    // FILL Qn00's USING Xnn0's
//    for (int n = 0; n < N; ++n) {
//        double xnn0 = sqrt(xnkl(n*N+n,0).real());
//        std::cout << xnn0 << std::endl;
//        qnlm(n, 0) = std::complex<double>(xnn0, 0.);
//    }
//    for (int n = 0; n < N; ++n) {
//        for (int l = 0; l <= L; ++l) {
//            double xnnl = sqrt(xnkl(n*N+n,l).real());
//            qnlm(n, l*l+l) = std::complex<double>(xnnl, 0.);
//        }
//    }

    return;
}

void AtomicSpectrum::addQnlm(std::string type, qnlm_t &nb_expansion) {
    assert(nb_expansion.getBasis() == _basis &&
        "Should not sum expansions linked against different bases.");
    map_qnlm_t::iterator it = _map_qnlm.find(type);
    if (it == _map_qnlm.end()) {
        _map_qnlm[type] = new BasisExpansion(_basis);
        it = _map_qnlm.find(type);
    }
    it->second->add(nb_expansion);
    _qnlm_generic->add(nb_expansion);
    return;
}

void AtomicSpectrum::addQnlmNeighbour(Particle *nb, qnlm_t *nb_expansion) {
    std::string type = nb->getType();
    int id = nb->getId();
    this->addQnlm(type, *nb_expansion);

    if (nb == this->getCenter()) {
        delete nb_expansion; // <- gradients should be zero, do not store
        //std::cout << "DO NOT STORE" << std::endl;
    }
    else {
        auto it = _map_pid_qnlm.find(id);
        if (it != _map_pid_qnlm.end()) {

            // There is already an entry for this pid - hence, this must be an image of the actual particle.
            // Add coefficients & gradients to existing density expansion.
            // This sanity check is no longer adequate:
            // throw soap::base::SanityCheckFailed("<AtomicSpectrum::addQnlm> Already have entry for pid.");
            _map_pid_qnlm[id].second->add(*nb_expansion);
            if (nb_expansion->hasGradients()) {
                _map_pid_qnlm[id].second->addGradient(*nb_expansion);
            }
        }
        else {
            _map_pid_qnlm[id] = std::pair<std::string,qnlm_t*>(type, nb_expansion);
        }
    }
    return;
}

void AtomicSpectrum::mergeQnlm(AtomicSpectrum *other, double scale, bool gradients) {
    // Function used to construct global spectrum as sum over atomic spectra.
    // The result is itself an "atomic" spectrum (as data fields are largely identical,
    // except for the fact that this summed spectrum does not have a well-defined center.
    // The <scale> factor modifies the qnlm, whereas their gradients remain unaltered.
    // For global spectrum, <scale> should be 0.5 in order to guarantee consistency
    // between the scalars qnlm and their gradients (or, in other words, the 0.5 ensures
    // that the global spectrum only counts all pairs once).
    assert(other->getBasis() == _basis &&
        "Should not merge atomic spectra linked against different bases.");
    // Type-agnostic (=generic) density expansion
    _qnlm_generic->add(*other->getQnlmGeneric(), scale);
    // Type-resolved (=specific) density expansions
    map_qnlm_t &map_qnlm_other = other->getQnlmMap();
    for (auto it = map_qnlm_other.begin(); it != map_qnlm_other.end(); ++it) {
        std::string density_type = it->first;
        BasisExpansion *density = it->second;
        // Already have density of this type?
        auto mit = _map_qnlm.find(density_type);
        if (mit == _map_qnlm.end()) {
            _map_qnlm[density_type] = new qnlm_t(_basis);
            mit = _map_qnlm.find(density_type);
        }
        // Add ...
        mit->second->add(*density, scale);
    }
    // Particle-ID-resolved gradients
    if (gradients) {
        map_pid_qnlm_t &map_pid_qnlm_other = other->getPidQnlmMap();
        for (auto it = map_pid_qnlm_other.begin(); it != map_pid_qnlm_other.end(); ++it) {
            int pid = it->first;
            std::string pid_type = it->second.first;
            qnlm_t *density_grad = it->second.second;
            // Already have density gradient for this particle?
            auto mit = _map_pid_qnlm.find(pid);
            if (mit == _map_pid_qnlm.end()) {
                qnlm_t *qnlm = new qnlm_t(_basis);
                // Remember to setup zero matrices to store gradient ...
                qnlm->zeroGradient();
                _map_pid_qnlm[pid] = std::pair<std::string,qnlm_t*>(pid_type, qnlm);
                mit = _map_pid_qnlm.find(pid);
            }
            // Add gradients ...
            mit->second.second->addGradient(*density_grad);
        }
    }
    return;
}

AtomicSpectrum::qnlm_t *AtomicSpectrum::getQnlm(std::string type) {
    if (type == "") {
        return _qnlm_generic;
    }
    map_qnlm_t::iterator it = _map_qnlm.find(type);
    if (it == _map_qnlm.end()) {
        throw soap::base::OutOfRange("AtomicSpectrum: No such type '" + type + "'");
    }
    else {
        return it->second;
    }
}

void AtomicSpectrum::computePowerGradients() {
    GLOG() << "CID " << _center_id << ": " << std::endl;
    for (auto it1 = _map_pid_qnlm.begin(); it1 != _map_pid_qnlm.end(); ++it1) {
        // Derivatives are taken with respect to the coordinates of this neighbour particle
        int pi_id = it1->first;
        std::string pi_type = it1->second.first;
        qnlm_t *pi_qnlm = it1->second.second;
        // Check and prepare storage in map
        auto it = _map_pid_xnkl.find(pi_id);
        if (it != _map_pid_xnkl.end()) {
            soap::base::SanityCheckFailed("<AtomicSpectrum::computePowerGradients> Already have entry for pid.");
        }
        _map_pid_xnkl[pi_id] = map_xnkl_t();
        map_xnkl_t &xnkl_map = _map_pid_xnkl[pi_id];
        // Compute for all type pairs
        GLOG() << "    NID " << pi_id << " " << std::flush;
        for (auto it2 = _map_qnlm.begin(); it2 != _map_qnlm.end(); ++it2) {
            std::string type_other = it2->first;
            qnlm_t *sum_qnlm_type_other = it2->second;
            if (pi_type == type_other) {
                // Type 1 == type 2
                GLOG() << pi_type << "=" << type_other << " " << std::flush;
                PowerExpansion *powex = new PowerExpansion(_basis);
                powex->computeCoefficientsGradients(pi_qnlm, sum_qnlm_type_other, true);
                // Store ...
                type_pair_t types(pi_type, type_other);
                auto it = xnkl_map.find(types);
                assert(it == xnkl_map.end());
                xnkl_map[types] = powex;
            }
            else {
                // Type 1 != type 2
                GLOG() << pi_type << ":" << type_other << " " << std::flush;
                PowerExpansion *powex_12 = new PowerExpansion(_basis);
                powex_12->computeCoefficientsGradients(pi_qnlm, sum_qnlm_type_other, false);
                GLOG() << type_other << ":" << pi_type << " " << std::flush;
                PowerExpansion *powex_21 = new PowerExpansion(_basis);
                powex_21->computeCoefficientsGradients(sum_qnlm_type_other, pi_qnlm, false);
                // Store ...
                type_pair_t types_12(pi_type, type_other);
                type_pair_t types_21(type_other, pi_type);
                auto it12 = xnkl_map.find(types_12);
                auto it21 = xnkl_map.find(types_21);
                assert(it12 == xnkl_map.end() && it21 == xnkl_map.end());
                xnkl_map[types_12] = powex_12;
                xnkl_map[types_21] = powex_21;
            }
        }
        if (_qnlm_generic) {
            auto it = _map_pid_xnkl_gc.find(pi_id);
            if (it != _map_pid_xnkl_gc.end()) assert(false && "Already have Xnkl_gc gradients for pid.");
            GLOG() << "*:* " << std::flush;
            PowerExpansion *powex = new PowerExpansion(_basis);
            powex->computeCoefficientsGradients(pi_qnlm, _qnlm_generic, true);
            _map_pid_xnkl_gc[pi_id] = powex;
        }
        GLOG() << std::endl;
    }
    return;
}

void AtomicSpectrum::computePower() {
    // TODO Calling this function more than once with the same object
    // TODO causes memory leaks => check for existing _map_xnkl entries
    // Specific (i.e., type-dependent)
    map_qnlm_t::iterator it1;
    map_qnlm_t::iterator it2;
    for (it1 = _map_qnlm.begin(); it1 != _map_qnlm.end(); ++it1) {
        for (it2 = _map_qnlm.begin(); it2 != _map_qnlm.end(); ++it2) {
            type_pair_t types(it1->first, it2->first);
            GLOG() << " " << types.first << ":" << types.second << std::flush;
            PowerExpansion *powex = new PowerExpansion(_basis);
            powex->computeCoefficients(it1->second, it2->second);
            _map_xnkl[types] = powex;
        }
    }
    // Generic coherent
    GLOG() << " g/c" << std::flush;
    if (_xnkl_generic_coherent) delete _xnkl_generic_coherent;
    _xnkl_generic_coherent = new PowerExpansion(_basis);
    _xnkl_generic_coherent->computeCoefficients(_qnlm_generic, _qnlm_generic);
    // Generic incoherent
    GLOG() << " g/i" << std::flush;
    if (_xnkl_generic_incoherent) delete _xnkl_generic_incoherent;
    _xnkl_generic_incoherent = new PowerExpansion(_basis);
    map_xnkl_t::iterator it;
    for (it = _map_xnkl.begin(); it != _map_xnkl.end(); ++it) {
        _xnkl_generic_incoherent->add(it->second);
    }
    GLOG() << std::endl;
}

void AtomicSpectrum::write(std::ostream &ofs) {
    throw soap::base::NotImplemented("AtomicSpectrum::write");
    map_qnlm_t::iterator it;
    for (it = _map_qnlm.begin(); it != _map_qnlm.end(); ++it) {
        std::cout << "TYPE" << it->first << std::endl;
        qnlm_t *qnlm = it->second;
        qnlm_t::coeff_t &coeff = qnlm->getCoefficients();
        int L = qnlm->getBasis()->getAngBasis()->L();
        int N = qnlm->getBasis()->getRadBasis()->N();
        for (int n = 0; n < N; ++n) {
            for (int l = 0; l < (L+1); ++l) {
                for (int m = -l; m <= l; ++l) {
                    std::cout << n << " " << l << " " << m << " " << coeff(n,l*l+l+m) << std::endl;
                }
            }
        }
    }
    return;
}

AtomicSpectrum::xnkl_t *AtomicSpectrum::getXnkl(type_pair_t &types) {
    map_xnkl_t::iterator it = _map_xnkl.find(types);
    if (it == _map_xnkl.end()) {
        if (types.first == "" and types.second == "") {
            return _xnkl_generic_coherent;
        }
        else {
            throw soap::base::OutOfRange("AtomicSpectrum: No such type pair '" + types.first + ":" + types.second + "'");
        }
    }
    return it->second;
}

AtomicSpectrum::xnkl_t *AtomicSpectrum::getPower(std::string type1, std::string type2) {
    type_pair_t types(type1, type2);
    return this->getXnkl(types);
}

boost::python::list AtomicSpectrum::getTypes() {
    boost::python::list types;
    map_qnlm_t::iterator it;
    for (it = _map_qnlm.begin(); it != _map_qnlm.end(); ++it) {
        std::string type = it->first;
        types.append(type);
    }
    return types;
}

boost::python::list AtomicSpectrum::getNeighbourPids() {
    boost::python::list pids;
    for (auto it = _map_pid_xnkl_gc.begin(); it != _map_pid_xnkl_gc.end(); ++it) {
        pids.append(it->first);
    }
    return pids;
}

void AtomicSpectrum::registerPython() {
    using namespace boost::python;

    // Does not work with boost::python ??
    //xnkl_t *(AtomicSpectrum::*getXnkl_string_string)(std::string, std::string) = &AtomicSpectrum::getPower;

    class_<AtomicSpectrum, AtomicSpectrum*>("AtomicSpectrum", init<>())
        .def(init<Particle*, Basis*>())
        .add_property("basis", make_function(&AtomicSpectrum::getBasis, ref_existing()))
        .def("addLinear", &AtomicSpectrum::addQnlm)
        .def("getLinear", &AtomicSpectrum::getQnlm, return_value_policy<reference_existing_object>())
        .def("computePower", &AtomicSpectrum::computePower)
        .def("getTypes", &AtomicSpectrum::getTypes)
        .def("getNeighbourPids", &AtomicSpectrum::getNeighbourPids)
        .def("getPower", &AtomicSpectrum::getPower, return_value_policy<reference_existing_object>())
        .def("getPowerGradGeneric", &AtomicSpectrum::getPowerGradGeneric, return_value_policy<reference_existing_object>())
        .def("getCenter", &AtomicSpectrum::getCenter, return_value_policy<reference_existing_object>())
        .def("getCenterId", &AtomicSpectrum::getCenterId)
        .def("getCenterType", &AtomicSpectrum::getCenterType, return_value_policy<reference_existing_object>())
        .def("getCenterPos", &AtomicSpectrum::getCenterPos, return_value_policy<reference_existing_object>());
}

}

