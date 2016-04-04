#ifndef _SOAP_ATOMICSPECTRUM_HPP
#define _SOAP_ATOMICSPECTRUM_HPP

#include <assert.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/base_object.hpp>

#include "base/logger.hpp"
#include "types.hpp"
#include "globals.hpp"
#include "options.hpp"
#include "structure.hpp"
#include "basis.hpp"
#include "power.hpp"


namespace soap {


class AtomicSpectrum : public std::map<std::string, BasisExpansion*>
{
public:
	typedef BasisExpansion qnlm_t;
	typedef PowerExpansion xnkl_t;
	typedef std::map<std::string, qnlm_t*> map_qnlm_t; // <- key: center type, e.g. 'C'
	typedef std::map<std::pair<std::string, std::string>, xnkl_t*> map_xnkl_t; // <- key: type string pair, e.g. ('C','H')

	AtomicSpectrum(Particle *center, Basis *basis) :
		_center(center),
		_center_pos(center->getPos()),
		_center_type(center->getType()),
		_basis(basis),
		_xnkl_generic_coherent(NULL),
		_xnkl_generic_incoherent(NULL) {
		_qnlm_generic = new BasisExpansion(_basis);
	}
	AtomicSpectrum() :
		_center(NULL),
		_center_pos(vec(0,0,0)),
		_center_type("?"),
		_basis(NULL),
		_qnlm_generic(NULL),
		_xnkl_generic_coherent(NULL),
		_xnkl_generic_incoherent(NULL) { ; }
    ~AtomicSpectrum() {
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
    // CENTER & BASIS METHODS
    Particle *getCenter() { return _center; }
	std::string &getCenterType() { return _center_type; }
	vec &getCenterPos() { return _center_pos; }
	Basis *getBasis() { return _basis; }
	// QNLM METHODS
    void addQnlm(std::string type, qnlm_t &nb_expansion) {
    	assert(nb_expansion.getBasis() == _basis &&
            "Should not sum expansions linked against different bases.");
    	iterator it = _map_qnlm.find(type);
    	if (it == _map_qnlm.end()) {
    		_map_qnlm[type] = new BasisExpansion(_basis);
    		it = _map_qnlm.find(type);
    	}
    	it->second->add(nb_expansion);
    	_qnlm_generic->add(nb_expansion);
    	return;
    }
    qnlm_t *getQnlmGeneric() { return _qnlm_generic; }
    qnlm_t *getQnlm(std::string type) {
    	if (type == "") {
    		return _qnlm_generic;
    	}
    	iterator it = _map_qnlm.find(type);
    	if (it == _map_qnlm.end()) {
    		throw soap::base::OutOfRange("AtomicSpectrum: No such type '" + type + "'");
    		return NULL;
    	}
    	else {
    		return it->second;
    	}
    }

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	// Center & basis
    	arch & _center;
    	arch & _center_pos;
    	arch & _center_type;
    	arch & _basis;
    	// Qnlm's
    	arch & _map_qnlm;
    	arch & _qnlm_generic;
    	// Xnkl's
    	arch & _map_xnkl;
    	arch & _xnkl_generic_coherent;
    	arch & _xnkl_generic_incoherent;
    	return;
    }
protected:
    // CENTER & BASIS LINKS
	Particle *_center;
	vec _center_pos;
	std::string _center_type;
	Basis *_basis;
	// DENSITY EXPANSION
	map_qnlm_t _map_qnlm;
	qnlm_t *_qnlm_generic;
	// POWER DENSITY EXPANSION
	map_xnkl_t _map_xnkl;
	xnkl_t *_xnkl_generic_coherent;
	xnkl_t *_xnkl_generic_incoherent;
};


}

#endif /* _SOAP_ATOMICSPECTRUM_HPP_ */
