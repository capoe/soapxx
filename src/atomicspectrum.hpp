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
	typedef std::pair<std::string, std::string> type_pair_t;

	AtomicSpectrum(Particle *center, Basis *basis);
	AtomicSpectrum() { this->null(); }
   ~AtomicSpectrum();
    void null();
    void write(std::ostream &ofs);
    void invert(map_xnkl_t &map_xnkl, xnkl_t *xnkl_generic_coherent, std::string type1, std::string type2);
    void invert(xnkl_t *xnkl, std::string type1, std::string type2);

    // CENTER & BASIS METHODS
    Particle *getCenter() { return _center; }
	std::string &getCenterType() { return _center_type; }
	vec &getCenterPos() { return _center_pos; }
	Basis *getBasis() { return _basis; }
	// QNLM METHODS
    void addQnlm(std::string type, qnlm_t &nb_expansion);
    qnlm_t *getQnlm(std::string type);
    qnlm_t *getQnlmGeneric() { return _qnlm_generic; }
    // XNKL METHODS
    void computePower();
    xnkl_t *getPower(std::string type1, std::string type2);
    xnkl_t *getXnkl(type_pair_t &types);
    map_xnkl_t &getXnklMap() { return _map_xnkl; }
    xnkl_t *getXnklGenericCoherent() { return _xnkl_generic_coherent; }
    xnkl_t *getXnklGenericIncoherent() { return _xnkl_generic_incoherent; }

    boost::python::list getTypes();
    static void registerPython();

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
