#ifndef _SOAP_ATOMICSPECTRUM_HPP
#define _SOAP_ATOMICSPECTRUM_HPP

#include <assert.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/base_object.hpp>

#include "soap/base/logger.hpp"
#include "soap/types.hpp"
#include "soap/globals.hpp"
#include "soap/options.hpp"
#include "soap/structure.hpp"
#include "soap/basis.hpp"
#include "soap/power.hpp"


namespace soap {


class AtomicSpectrum : public std::map<std::string, BasisExpansion*>
{
public:
    // EXPANSION TYPES
	typedef BasisExpansion qnlm_t;
	typedef PowerExpansion xnkl_t;
	typedef std::pair<std::string, std::string> type_pair_t;

	// CONTAINERS FOR STORING SCALAR FIELDS
	typedef std::map<std::string, qnlm_t*> map_qnlm_t; // <- key: center type, e.g. 'C'
	typedef std::map<std::pair<std::string, std::string>, xnkl_t*> map_xnkl_t; // <- key: type string pair, e.g. ('C','H')

	// CONTAINERS FOR STORING GRADIENTS
	typedef std::map<int, std::pair<std::string,qnlm_t*> > map_pid_qnlm_t; // <- id=>(type;qnlm)
	typedef std::map<int, map_xnkl_t> map_pid_xnkl_t; // <- id=>type=>xnkl
	typedef std::map<int, xnkl_t*> map_pid_xnkl_gc_t; // <- id=>xnkl_generic_coherent


	AtomicSpectrum(Particle *center, Basis *basis);
	AtomicSpectrum(Basis *basis);
	AtomicSpectrum() { this->null(); }
   ~AtomicSpectrum();
    void null();
    void write(std::ostream &ofs);
    void invert(map_xnkl_t &map_xnkl, xnkl_t *xnkl_generic_coherent, std::string type1, std::string type2);
    void invert(xnkl_t *xnkl, std::string type1, std::string type2);

    // CENTER & BASIS METHODS
    Particle *getCenter() { return _center; }
    int getCenterId() { return _center_id; }
	std::string &getCenterType() { return _center_type; }
	vec &getCenterPos() { return _center_pos; }
	Basis *getBasis() { return _basis; }
	// QNLM METHODS
    void addQnlm(std::string type, qnlm_t &nb_expansion);
    void addQnlmNeighbour(Particle *nb, qnlm_t *nb_expansion);
    qnlm_t *getQnlm(std::string type);
    qnlm_t *getQnlmGeneric() { return _qnlm_generic; }
    map_qnlm_t &getQnlmMap() { return _map_qnlm; }
    map_pid_qnlm_t &getPidQnlmMap() { return _map_pid_qnlm; }
    void mergeQnlm(AtomicSpectrum *other, double scale, bool gradients);
    // XNKL METHODS
    void computePower();
    void computePowerGradients();
    xnkl_t *getPower(std::string type1, std::string type2);
    xnkl_t *getPowerGradGeneric(int pid) { return _map_pid_xnkl_gc[pid]; }
    xnkl_t *getXnkl(type_pair_t &types);
    map_xnkl_t &getXnklMap() { return _map_xnkl; }
    xnkl_t *getXnklGenericCoherent() { return _xnkl_generic_coherent; }
    xnkl_t *getXnklGenericIncoherent() { return _xnkl_generic_incoherent; }

    boost::python::list getTypes();
    boost::python::list getNeighbourPids();
    static void registerPython();

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	// Center & basis
    	arch & _center;
    	arch & _center_id;
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
    	// PID-resolved
    	arch & _map_pid_qnlm;
    	arch & _map_pid_xnkl;
    	arch & _map_pid_xnkl_gc;
    	return;
    }
protected:
    // CENTER & BASIS LINKS
	Particle *_center;
	int _center_id;
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

	// PID-RESOLVED (GRADIENTS)
	map_pid_qnlm_t _map_pid_qnlm; // <- for gradients wrt position of neighbour with global id
	map_pid_xnkl_t _map_pid_xnkl;
	map_pid_xnkl_gc_t _map_pid_xnkl_gc;
};


}

#endif /* _SOAP_ATOMICSPECTRUM_HPP_ */
