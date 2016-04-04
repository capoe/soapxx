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

AtomicSpectrum::xnkl_t *AtomicSpectrum::getXnkl(type_pair_t &types) {
	map_xnkl_t::iterator it = _map_xnkl.find(types);
	if (it == _map_xnkl.end()) {
		throw soap::base::OutOfRange("AtomicSpectrum: No such type pair '" + types.first + ":" + types.second + "'");
	}
	return it->second;
}

}

