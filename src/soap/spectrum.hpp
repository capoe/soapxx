#ifndef _SOAP_SPECTRUM_HPP
#define _SOAP_SPECTRUM_HPP

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
#include "soap/atomicspectrum.hpp"
//#include "soap/linalg/Eigen/Dense"


namespace soap {

namespace ub = boost::numeric::ublas;
namespace bpy = boost::python;

class Spectrum
{
  public:
	typedef std::vector<AtomicSpectrum*> atomspec_array_t;
	typedef std::vector<AtomicSpectrum*>::iterator atomic_it_t;
	typedef std::map<std::string, atomspec_array_t> map_atomspec_array_t;

    Spectrum();
	Spectrum(std::string archfile);
	Spectrum(Structure &structure, Options &options);
	Spectrum(Structure &structure, Options &options, Basis &basis);
   ~Spectrum();

    Structure *getStructure() { return _structure; }
    Options *getOptions() { return _options; }
    Basis *getBasis() { return _basis; }

    atomic_it_t beginAtomic() { return _atomspec_array.begin(); }
    atomic_it_t endAtomic() { return _atomspec_array.end(); }

	void saveAndClean() { std::cout << "spectrum::save&clean" << std::endl; }
	void save(std::string archfile);
    std::string saves(bool prune = true);
	void load(std::string archfile);
    Spectrum &loads(std::string bstr);
	void clean();
    int length() { return _atomspec_array.size(); }

	void compute();
	void compute2D();
    void compute(Segment *centers);
	void compute(Segment *centers, Segment *targets);
	void compute(Structure::particle_array_t &sources, Structure::particle_array_t &targets);
	void compute2D(Structure::particle_array_t &sources, Structure::particle_array_t &targets, Structure::laplace_t &L);
	AtomicSpectrum *computeAtomic(Particle *center);
	AtomicSpectrum *computeAtomic(Particle *center, Structure::particle_array_t &targets);
	AtomicSpectrum *computeAtomic2D(Particle *center, Structure::particle_array_t &targets, Structure::laplace_t &L);
    AtomicSpectrum *computeGlobal();
	void deleteGlobal() { if (_global_atomic) { delete _global_atomic; _global_atomic = NULL; } }
	void addAtomic(AtomicSpectrum *atomspec);
	AtomicSpectrum *getAtomic(int slot_idx, std::string center_type);
	AtomicSpectrum *getGlobal() { assert(_global_atomic && "Compute first"); return _global_atomic; }
	void writeDensityOnGrid(int slot_idx, std::string center_type, std::string density_type);
    void writeDensityCubeFile(int atom_idx, std::string density_type, std::string filename, bool from_expansion);
	void writeDensityOnGridInverse(int slot_idx, std::string center_type, std::string type1, std::string type2);
	void writeDensity(int slot_idx, std::string center_type, std::string density_type);
	void writePowerDensity(int slot_idx, std::string center_type, std::string type1, std::string type2);

	void computePower();
	void computePowerGradients();
	void computeLinear();

	static void registerPython();

	template<class Archive>
	void serialize(Archive &arch, const unsigned int version) {
		arch & _options;
		arch & _structure;
		arch & _basis;

		arch & _atomspec_array;
		arch & _map_atomspec_array;

		arch & _global_atomic;
		return;
	}

  private:

	Logger *_log;
	Options *_options;
    Structure *_structure;
    Basis *_basis;
    bool _own_basis;

    atomspec_array_t _atomspec_array;
    map_atomspec_array_t _map_atomspec_array;

    AtomicSpectrum *_global_atomic;
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


#endif /* _SOAP_SPECTRUM_HPP_ */
