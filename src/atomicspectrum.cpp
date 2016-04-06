#include <fstream>
#include <boost/format.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "atomicspectrum.hpp"

namespace soap {

AtomicSpectrum::AtomicSpectrum(Particle *center, Basis *basis) {
	this->null();
	_center = center;
	_center_pos = center->getPos();
	_center_type = center->getType();
	_basis = basis;
	_qnlm_generic = new BasisExpansion(_basis);
}

AtomicSpectrum::~AtomicSpectrum() {
	// Clean Qnlm's
	for (map_qnlm_t::iterator it = _map_qnlm.begin(); it != _map_qnlm.end(); ++it) delete it->second;
	_map_qnlm.clear();
	if (_qnlm_generic) {
		delete _qnlm_generic;
		_qnlm_generic = NULL;
	}
	// Clean Xnkl's
	for (map_xnkl_t::iterator it = _map_xnkl.begin(); it != _map_xnkl.end(); ++it) delete it->second;
	_map_xnkl.clear();
	if (_xnkl_generic_coherent) {
		delete _xnkl_generic_coherent;
		_xnkl_generic_coherent = NULL;
	}
	if (_xnkl_generic_incoherent) {
		delete _xnkl_generic_incoherent;
		_xnkl_generic_incoherent = NULL;
	}
}

void AtomicSpectrum::null() {
	_center = NULL;
	_center_pos = vec(0,0,0);
	_center_type = "?";
	_basis = NULL;
	_qnlm_generic = NULL;
	_xnkl_generic_coherent = NULL;
	_xnkl_generic_incoherent = NULL;
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
    qnlm(n, l*l+l+m) = cmplx(sqrt(xnkl(n*N+k, l).real()), 0.);
    for (k = 1; k < N; ++k) {
    	qnlm(k, l*l+l+m) = cmplx(xnkl(n*N+k, l).real()/qnlm(0, l*l+l+m).real(), 0.);
    }

//    // FILL Qn00's USING Xnn0's
//    for (int n = 0; n < N; ++n) {
//    	double xnn0 = sqrt(xnkl(n*N+n,0).real());
//    	std::cout << xnn0 << std::endl;
//    	qnlm(n, 0) = std::complex<double>(xnn0, 0.);
//    }
//    for (int n = 0; n < N; ++n) {
//        for (int l = 0; l <= L; ++l) {
//        	double xnnl = sqrt(xnkl(n*N+n,l).real());
//        	qnlm(n, l*l+l) = std::complex<double>(xnnl, 0.);
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

void AtomicSpectrum::computePower() {
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
		if (types.first == "g" and types.second == "c") {
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

void AtomicSpectrum::registerPython() {
    using namespace boost::python;

    // Does not work with boost::python ??
    //xnkl_t *(AtomicSpectrum::*getXnkl_string_string)(std::string, std::string) = &AtomicSpectrum::getPower;

    class_<AtomicSpectrum, AtomicSpectrum*>("AtomicSpectrum", init<>())
        .def(init<Particle*, Basis*>())
		.def("addLinear", &AtomicSpectrum::addQnlm)
		.def("getLinear", &AtomicSpectrum::getQnlm, return_value_policy<reference_existing_object>())
		.def("computePower", &AtomicSpectrum::computePower)
    	.def("getPower", &AtomicSpectrum::getPower, return_value_policy<reference_existing_object>())
	    .def("getCenter", &AtomicSpectrum::getCenter, return_value_policy<reference_existing_object>())
		.def("getCenterType", &AtomicSpectrum::getCenterType, return_value_policy<reference_existing_object>())
	    .def("getCenterPos", &AtomicSpectrum::getCenterPos, return_value_policy<reference_existing_object>());
}

}

