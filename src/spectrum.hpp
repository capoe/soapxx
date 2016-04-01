#ifndef _SOAP_SPECTRUM_HPP
#define _SOAP_SPECTRUM_HPP

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


namespace soap {


class AtomicSpectrum : public std::map<std::string, BasisExpansion*>
{
public:
	AtomicSpectrum(Particle *center, Basis *basis) :
		_center(center),
		_center_pos(center->getPos()),
		_center_type(center->getType()),
		_basis(basis) {
		_expansion_reduced = new BasisExpansion(_basis);
	}
	AtomicSpectrum() :
		_center(NULL),
		_center_pos(vec(0,0,0)),
		_center_type("?"),
		_basis(NULL),
		_expansion_reduced(NULL) { ; }
    ~AtomicSpectrum() {
        iterator it;
        for (it = this->begin(); it != this->end(); ++it) delete it->second;
        this->clear();
        delete _expansion_reduced;
    }
    void add(std::string type, BasisExpansion &nb_expansion) {
    	assert(nb_expansion.getBasis() == _basis &&
            "Should not sum expansions linked against different bases.");
    	iterator it = this->find(type);
    	if (it == this->end()) {
    		(*this)[type] = new BasisExpansion(_basis);
    		it = this->find(type);
    	}
    	it->second->add(nb_expansion);
    	_expansion_reduced->add(nb_expansion);
    	return;
    }
    Particle *getCenter() { return _center; }
    std::string &getCenterType() { return _center_type; }
    vec &getCenterPos() { return _center_pos; }
    BasisExpansion *getReduced() { return _expansion_reduced; }
    BasisExpansion *getExpansion(std::string type) {
    	if (type == "") {
    		return _expansion_reduced;
    	}
    	iterator it = this->find(type);
    	if (it == this->end()) {
    		throw soap::base::OutOfRange("AtomicSpectrum: No such type '" + type + "'");
    		return NULL;
    	}
    	else {
    		return it->second;
    	}
    }
    Basis *getBasis() { return _basis; }

    template<class Archive>
    void serialize(Archive &arch, const unsigned int version) {
    	arch & _center;
    	arch & _center_pos;
    	arch & _center_type;
    	arch & _basis;
    	arch & _expansion_reduced;
    	return;
    }
protected:
	Particle *_center;
	vec _center_pos;
	std::string _center_type;
	Basis *_basis;
	BasisExpansion *_expansion_reduced;
	//std::map<std::string, BasisExpansion*> _map_type_expansion;
};


class CenterDensity
{
	CenterDensity() {}
};


class TargetDensity
{
	TargetDensity() {}
};

class Center
{
	Center() {}
};

class Target
{
	Target() {}
};

// Need this: Spectrum(System1, System2, options) where Sys1 <> Sources, Sys2 <> Targets


class Spectrum
{
public:
	typedef std::vector<AtomicSpectrum*> atomspec_array_t;
	typedef std::map<std::string, atomspec_array_t> map_atomspec_array_t;

	Spectrum(std::string archfile);
	Spectrum(Structure &structure, Options &options);
   ~Spectrum();

	void saveAndClean() { std::cout << "spectrum::save&clean" << std::endl; }
	void save(std::string archfile);
	void load(std::string archfile);
	void clean();

	void compute();
	AtomicSpectrum *computeAtomic(Particle *center);
	void add(AtomicSpectrum *atomspec);
	void writeDensityOnGrid(int slot_idx, std::string center_type, std::string density_type);

	void computePower();
	void computeLinear();

	static void registerPython();

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & _options;
		arch & _structure;
		arch & _basis;

		arch & _atomspec_array;
		arch & _map_atomspec_array;
		return;
	}

private:

	Logger *_log;
	Options *_options;
    Structure *_structure;
    Basis *_basis;

    atomspec_array_t _atomspec_array;
    map_atomspec_array_t _map_atomspec_array;
};






class PairSpectrum
{
public:
	PairSpectrum(
	    Structure &struct1,
        Structure &struct2,
		Options &options)
        : _struct1(&struct1), _struct2(&struct2), _options(&options) {}

	void compute() {
        return;
	}
	void saveAndClean() { std::cout << "pairspectrum::save&clean" << std::endl; }
	static void registerPython() {
		using namespace boost::python;
		class_<PairSpectrum>("PairSpectrum",
		    init<Structure &, Structure &, Options &>())
			.def("compute", &PairSpectrum::compute)
			.def("saveAndClean", &PairSpectrum::saveAndClean);
	}

private:
    Structure *_struct1;
	Structure *_struct2;
	Options *_options;
};

class SpectrumOverlap
{
	SpectrumOverlap() {}
};





/* STORAGE, BASIS, COMPUTATION, PARALLELIZATION */
/*
 *
 * Spectrum, PowerSpectrum
 * SpectrumK, PowerSpectrumK
 * StructureK, ParticleK (with weight, type)
 *
 * Basis
 * BasisFactory
 *
 * RadialBasis
 * RadialBasisLegendre
 * RadialBasisGaussian
 * RadialBasisHermite
 *
 * AngularBasis
 * AngularBasisHarmonic
 */


/*
 * Parallelization
 *     based on centers
 *     based on wavevectors
 */

/*
 * Spectrum->Setup(system, options)
 *     -> Basis->Setup()
 * Compute(patterns)
 *     for pattern in patterns:
 *        single out neighbours that match pattern
 *        for each neighbour:
 *            -> Compute(pos, refpos)
 */

/*
 * PowerSpectrum->Setup(Expansion)
 *      [a][b][n][n'][l] = sum{m} [a][nlm]*[b][n'lm]
 *
 * Spectrum in linear storage:
 * Start n at offset_n = (n-1)*(L+1)^2
 *     Start l at offset_l = offset_n + l^2
 *         Start m at offset_m = offset_l + (m+l)
 *
 * PowerSpectrum in linear storage:
 * Start n at offset_n = (n-1)*N*(L+1)
 *     Start n' at offset_n' = offset_n + (n'-1)*(L+1)
 *         Start l at offset_l = offset_n'
 *
 * map< pattern1, spectrum  >
 * for (nit in spectrum)
 *     for (lit in nit)
 *         for (mit in lit)
 *             nit->n
 *             lit->l
 *             mit->m
 *
 *
 */

/*
 * Storage:
 *     -> Serialize()
 *     For each center:
 *         for each pattern:
 *             [n][l][m]
 *     Store [n] and [l][m] separately?
 *
 *     -> PowerSerialize()
 *     For each center:
 *         for each pair of patterns:
 *             [n][n'][l]
 */

/*
 *
 * Types of basis functions: Legendre, Hermite, Gaussian*
 * Make sure to include appropriate n-factors, e.g. sqrt(2*n+1) for Legendre
 * Normalized on which interval?
 *
 * e.g. RadialBasisLegendre : RadialBasis
 *
 * RadialBasis->Setup()
 *     -> Orthogonalize()
 * RadialBasis -> EvaluateAt(radius)
 *     -> AtomicBasis->EvaluateAt(radius)
 *     -> Transform coefficients
 *     -> Return coefficients (N-vector)
 * RadialBasis->Iterator() ?
 *
 */

/*
 * AngularBasis->Setup()
 * AngularBasis->EvaluateAt(direction)
 *     -> AtomicBasisL->EvaluateAt(radius)
 *          -> AtomicBasisLM -> EvaluateAt(radius)
 *     -> Return coefficients (L*M-vector)
 * AngularBasis->Iterator()
 */



}


#endif /* _SOAP_RADIALBASIS_HPP_ */
