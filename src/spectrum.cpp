#include <fstream>
#include <boost/format.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "spectrum.hpp"

namespace soap {

Spectrum::Spectrum(Structure &structure, Options &options) :
    _log(NULL), _options(&options), _structure(&structure), _own_basis(true) {
	GLOG() << "Configuring spectrum ..." << std::endl;
	// CREATE & CONFIGURE BASIS
	_basis = new Basis(&options);
}

Spectrum::Spectrum(Structure &structure, Options &options, Basis &basis) :
	_log(NULL), _options(&options), _structure(&structure), _basis(&basis), _own_basis(false) {
	;
}

Spectrum::Spectrum(std::string archfile) :
	_log(NULL), _options(NULL), _structure(NULL), _basis(NULL), _own_basis(true) {
	this->load(archfile);
}

Spectrum::~Spectrum() {
	delete _log;
	_log = NULL;
	if (_own_basis) {
		delete _basis;
		_basis = NULL;
	}
	atomspec_array_t::iterator it;
	for (it = _atomspec_array.begin(); it != _atomspec_array.end(); ++it) {
		delete *it;
	}
	_atomspec_array.clear();
	// AtomicSpectra in type map already deleted above, clear only:
	_map_atomspec_array.clear();
}

void Spectrum::compute() {
	GLOG() << "Compute spectrum ..." << std::endl;
    GLOG() << _options->summarizeOptions() << std::endl;
    GLOG() << "Using radial basis of type '" << _basis->getRadBasis()->identify() << "'" << std::endl;
    GLOG() << "Using angular basis of type '" << _basis->getAngBasis()->identify() << "'" << std::endl;
    GLOG() << "Using cutoff function of type '" << _basis->getCutoff()->identify() << "'" << std::endl;

    Structure::particle_it_t pit;
    for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
    	AtomicSpectrum *atomic_spectrum = this->computeAtomic(*pit);
    	//atomic_spectrum->getReduced()->writeDensityOnGrid("density.expanded.cube", _options, _structure, *pit, true);
    	//atomic_spectrum->getReduced()->writeDensityOnGrid("density.explicit.cube", _options, _structure, *pit, false);
    	this->addAtomic(atomic_spectrum);
    }
}

AtomicSpectrum *Spectrum::getAtomic(int slot_idx, std::string center_type) {
	AtomicSpectrum *atomic_spectrum = NULL;
	// FIND SPECTRUM
	if (center_type == "") {
		// NO TYPE => ACCESS ARRAY
		// Slot index valid?
		if (slot_idx < _atomspec_array.size()) {
			atomic_spectrum = _atomspec_array[slot_idx];
		}
		else {
			throw soap::base::OutOfRange("Spectrum slot index");
		}
	}
	else {
		// TYPE => ACCESS TYPE MAP
		map_atomspec_array_t::iterator it = _map_atomspec_array.find(center_type);
		// Any such type?
		if (it == _map_atomspec_array.end()) {
			throw soap::base::OutOfRange("No spectrum of type '" + center_type + "'");
		}
		else {
			// Slot index valid?
			if (slot_idx < it->second.size()) {
				atomic_spectrum = it->second[slot_idx];
			}
			else {
				throw soap::base::OutOfRange("Spectrum slot index, type '" + center_type + "'");
			}
		}
	}
	return atomic_spectrum;
}

void Spectrum::writeDensityOnGrid(int slot_idx, std::string center_type, std::string density_type) {
	AtomicSpectrum *atomic_spectrum = this->getAtomic(slot_idx, center_type);
	// WRITE CUBE FILES
	if (atomic_spectrum) {
		atomic_spectrum->getQnlm(density_type)->writeDensityOnGrid(
			"density.expanded.cube", _options, _structure, atomic_spectrum->getCenter(), true);
		atomic_spectrum->getQnlm(density_type)->writeDensityOnGrid(
			"density.explicit.cube", _options, _structure, atomic_spectrum->getCenter(), false);
	}
	return;
}

void Spectrum::writeDensityOnGridInverse(int slot_idx, std::string center_type, std::string type1, std::string type2) {
	AtomicSpectrum *atomic_spectrum = this->getAtomic(slot_idx, center_type);
	AtomicSpectrum inverse_atomic_spectrum(atomic_spectrum->getCenter(), atomic_spectrum->getBasis());
	inverse_atomic_spectrum.invert(atomic_spectrum->getXnklMap(), atomic_spectrum->getXnklGenericCoherent(), type1, type2);
	inverse_atomic_spectrum.getQnlm("")->writeDensityOnGrid(
	    "density.inverse.cube", _options, _structure, inverse_atomic_spectrum.getCenter(), true);
	inverse_atomic_spectrum.getQnlm("")->writeDensity(
		"density.inverse.coeff", _options, _structure, inverse_atomic_spectrum.getCenter());
	return;
}

void Spectrum::writeDensity(int slot_idx, std::string center_type, std::string density_type) {
	AtomicSpectrum *atomic_spectrum = this->getAtomic(slot_idx, center_type);
	// WRITE COEFF FILE
	if (atomic_spectrum) {
		atomic_spectrum->getQnlm(density_type)->writeDensity(
			"density.expanded.coeff", _options, _structure, atomic_spectrum->getCenter());
	}
	return;
}

void Spectrum::writePowerDensity(int slot_idx, std::string center_type, std::string type1, std::string type2) {
	AtomicSpectrum *atomic_spectrum = this->getAtomic(slot_idx, center_type);
	if (atomic_spectrum) {
		AtomicSpectrum::type_pair_t types(type1, type2);
		atomic_spectrum->getXnkl(types)->writeDensity(
			"density.power.coeff", _options, _structure, atomic_spectrum->getCenter());
	}
	return;
}

void Spectrum::computePower() {
	atomspec_array_t::iterator it;
	for (it = _atomspec_array.begin(); it != _atomspec_array.end(); ++it) {
		(*it)->computePower();
	}
	return;
}

// TODO change to ::computeAtomic(Particle *center, vector<Particle*> &nbhood)
AtomicSpectrum *Spectrum::computeAtomic(Particle *center) {
	GLOG() << "Compute atomic spectrum for particle " << center->getId()
	    << " (type " << center->getType() << ")" << std::endl;

//    RadialCoefficients c_n_zero = _radbasis->computeCoefficientsAllZero();
//    AngularCoefficients c_lm_zero = _angbasis->computeCoefficientsAllZero();
//    BasisCoefficients c_nlm(c_n_zero, c_lm_zero);
//    c_nlm.linkBasis(_radbasis, _angbasis);

//    BasisExpansion *nbhood_expansion = new BasisExpansion(this->_basis);
    AtomicSpectrum *atomic_spectrum = new AtomicSpectrum(center, this->_basis);

    Structure::particle_it_t pit;
    for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {

    	// FIND DISTANCE & DIRECTION, APPLY CUTOFF (= WEIGHT REDUCTION)
    	vec dr = _structure->connect(center->getPos(), (*pit)->getPos());
    	double r = soap::linalg::abs(dr);
    	double weight_scale = this->_basis->getCutoff()->calculateWeight(r);
    	if (weight_scale < 0.) continue; // <- Negative cutoff weight means: skip
    	vec d = dr/r;

    	// APPLY WEIGHT IF CENTER
    	if (*pit == center) {
			weight_scale *= this->_basis->getCutoff()->getCenterWeight();
		}

    	// COMPUTE EXPANSION & ADD TO SPECTRUM
    	BasisExpansion nb_expansion(this->_basis);
    	nb_expansion.computeCoefficients(r, d, weight_scale*(*pit)->getWeight(), (*pit)->getSigma());
    	std::string type_other = (*pit)->getType();
    	atomic_spectrum->addQnlm(type_other, nb_expansion);

//    	nbhood_expansion->add(nb_expansion);


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

//    nbhood_expansion->writeDensityOnGrid("density.expanded.cube", _options, _structure, center, true);
//    nbhood_expansion->writeDensityOnGrid("density.explicit.cube", _options, _structure, center, false);



	return atomic_spectrum;
}


void Spectrum::addAtomic(AtomicSpectrum *atomspec) {
	assert(atomspec->getBasis() == _basis &&
		"Should not append atomic spectrum linked against different basis.");
	std::string atomspec_type = atomspec->getCenterType();
	map_atomspec_array_t::iterator it = _map_atomspec_array.find(atomspec_type);
	if (it == _map_atomspec_array.end()) {
		_map_atomspec_array[atomspec_type] = atomspec_array_t();
		it = _map_atomspec_array.find(atomspec_type);
	}
	it->second.push_back(atomspec);
	_atomspec_array.push_back(atomspec);
	return;
}

void Spectrum::save(std::string archfile) {
	std::ofstream ofs(archfile.c_str());
	boost::archive::binary_oarchive arch(ofs);
	arch << (*this);
	return;
}

void Spectrum::load(std::string archfile) {
	std::ifstream ifs(archfile.c_str());
	boost::archive::binary_iarchive arch(ifs);
	arch >> (*this);
	return;
}

void Spectrum::registerPython() {
    using namespace boost::python;
    class_<Spectrum>("Spectrum", init<Structure &, Options &>())
    	.def(init<Structure &, Options &, Basis &>())
    	.def(init<std::string>())
	    .def("compute", &Spectrum::compute)
		.def("computePower", &Spectrum::computePower)
		.def("addAtomic", &Spectrum::addAtomic)
		.def("getAtomic", &Spectrum::getAtomic, return_value_policy<reference_existing_object>())
	    .def("saveAndClean", &Spectrum::saveAndClean)
		.def("save", &Spectrum::save)
		.def("load", &Spectrum::load)
        .def("writeDensityOnGrid", &Spectrum::writeDensityOnGrid)
		.def("writeDensity", &Spectrum::writeDensity)
		.def("writePowerDensity", &Spectrum::writePowerDensity)
		.def("writeDensityOnGridInverse", &Spectrum::writeDensityOnGridInverse)
		.add_property("options", make_function(&Spectrum::getOptions, ref_existing()))
		.add_property("basis", make_function(&Spectrum::getBasis, ref_existing()))
		.add_property("structure", make_function(&Spectrum::getStructure, ref_existing()));
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

