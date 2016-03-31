#include <fstream>
#include <boost/format.hpp>

#include "spectrum.hpp"

namespace soap {

Spectrum::Spectrum(Structure &structure, Options &options)
    : _log(NULL), _options(&options), _structure(&structure) {
	GLOG() << "Configuring spectrum ..." << std::endl;
	// CREATE & CONFIGURE BASIS
	_basis = new Basis(&options);
}

Spectrum::~Spectrum() {
	delete _log;
	_log = NULL;
	delete _basis;
	_basis = NULL;
}

void Spectrum::compute() {
	GLOG() << "Compute spectrum ..." << std::endl;
    GLOG() << _options->summarizeOptions() << std::endl;
    GLOG() << "Using radial basis of type '" << _basis->getRadBasis()->identify() << "'" << std::endl;
    GLOG() << "Using angular basis of type '" << _basis->getAngBasis()->identify() << "'" << std::endl;

    Structure::particle_it_t pit;
    for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
    	this->computeAtomic(*pit);
    	// TODO Receive & store atomic spectrum
    	break;
    }
}

// TODO change to ::computeAtomic(Particle *center, vector<Particle*> &nbhood)
void Spectrum::computeAtomic(Particle *center) {
	GLOG() << "Compute atomic spectrum for particle " << center->getId()
	    << " (type " << center->getType() << ")" << std::endl;

	std::string type_this = center->getType();


//    RadialCoefficients c_n_zero = _radbasis->computeCoefficientsAllZero();
//    AngularCoefficients c_lm_zero = _angbasis->computeCoefficientsAllZero();
//    BasisCoefficients c_nlm(c_n_zero, c_lm_zero);
//    c_nlm.linkBasis(_radbasis, _angbasis);


    BasisExpansion *nbhood_expansion = new BasisExpansion(this->_basis);

    Structure::particle_it_t pit;
    for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {

    	// TODO Cut-off check && cut-off smoothing (= weight reduction)
    	vec dr = _structure->connect(center->getPos(), (*pit)->getPos());
    	double r = soap::linalg::abs(dr);
    	vec d = dr/r;
    	std::string type_other = (*pit)->getType();

    	BasisExpansion nb_expansion(this->_basis);
    	nb_expansion.computeCoefficients(r, d, (*pit)->getWeight(), (*pit)->getSigma());
    	nbhood_expansion->add(nb_expansion);


//        // COMPUTE RADIAL COEFFICIENTS
//    	RadialCoefficients c_n_pair = _radbasis->computeCoefficients(r);
//    	/*
//    	GLOG() << "Radial coefficients r = " << r << std::endl;
//        for (int i = 0; i < c_n.size(); ++i) {
//            GLOG() << c_n[i] << " | ";
//        }
//        GLOG() << std::endl;
//        */
//
//        // COMPUTE ANGULAR COEFFICIENTS
//        AngularCoefficients c_lm_pair = _angbasis->computeCoefficients(d, r);
//        /*
//    	GLOG() << "Angular coefficients d = " << d << std::endl;
//		for (int lm = 0; lm < c_lm.size(); ++lm) {
//			GLOG() << c_lm[lm] << " | ";
//		}
//		GLOG() << std::endl;
//		*/
//
//        BasisCoefficients c_nlm_pair(c_n_pair, c_lm_pair);
//        c_nlm.add(c_nlm_pair);


    }


//    c_nlm.writeDensityOnGrid("density.expanded.cube", _options, _structure, center, true);
//    c_nlm.writeDensityOnGrid("density.explicit.cube", _options, _structure, center, false);

    nbhood_expansion->writeDensityOnGrid("density.expanded.cube", _options, _structure, center, true);
    nbhood_expansion->writeDensityOnGrid("density.explicit.cube", _options, _structure, center, false);

    delete nbhood_expansion;
	return;
}

void Spectrum::registerPython() {
    using namespace boost::python;
    class_<Spectrum>("Spectrum", init<Structure &, Options &>())
	    .def("compute", &Spectrum::compute)
	    .def("saveAndClean", &Spectrum::saveAndClean);
}

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

