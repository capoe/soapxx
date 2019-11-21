#include <fstream>
#include <boost/format.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "soap/spectrum.hpp"
#include "soap/linalg/numpy.hpp"

namespace soap {

Spectrum::Spectrum(Structure &structure, Options &options) :
        _log(NULL), _options(&options), _structure(&structure), 
        _own_basis(true), _global_atomic(NULL) {
	GLOG() << "Configuring spectrum ..." << std::endl;
	_basis = new Basis(&options);
}

Spectrum::Spectrum(Structure &structure, Options &options, Basis &basis) :
        _log(NULL), _options(&options), _structure(&structure), 
        _basis(&basis), _own_basis(false), _global_atomic(NULL) {
	;
}

Spectrum::Spectrum(std::string archfile) :
        _log(NULL), _options(NULL), _structure(NULL), 
        _basis(NULL), _own_basis(true), _global_atomic(NULL) {
	this->load(archfile);
}

Spectrum::Spectrum() :
        _log(NULL), _options(NULL), _structure(NULL), 
        _basis(NULL), _own_basis(true), _global_atomic(NULL) {
    ;
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

	if (_global_atomic) delete _global_atomic;
	_global_atomic = NULL;
}

void Spectrum::compute() {
    if (_options->get<bool>("spectrum.2d"))
        this->compute2D();
    else
        this->compute(_structure->particles(), _structure->particles());
}

void Spectrum::compute(Segment *center) {
    this->compute(center->particles(), _structure->particles());
}

void Spectrum::compute(Segment *center, Segment *target) {
    this->compute(center->particles(), target->particles());
}

void Spectrum::compute(
        Structure::particle_array_t &centers, 
        Structure::particle_array_t &targets) {
    GLOG() << "Compute spectrum "
        << "(centers " << centers.size() << ", targets " << targets.size() << ") ..." << std::endl;
    GLOG() << _options->summarizeOptions() << std::endl;
    GLOG() << "Using radial basis of type '" << _basis->getRadBasis()->identify() << "'" << std::endl;
    GLOG() << "Using angular basis of type '" << _basis->getAngBasis()->identify() << "'" << std::endl;
    GLOG() << "Using cutoff function of type '" << _basis->getCutoff()->identify() << "'" << std::endl;

    Structure::particle_it_t pit;
    for (pit = centers.begin(); pit != centers.end(); ++pit) {
        // Continue if exclusion defined ...
        if (_options->doExcludeCenter((*pit)->getType()) ||
            _options->doExcludeCenterId((*pit)->getId())) continue;
        // Compute ...
        AtomicSpectrum *atomic_spectrum = this->computeAtomic(*pit, targets);
        this->addAtomic(atomic_spectrum);
    }
}

void Spectrum::computePower() {
    atomspec_array_t::iterator it;
    for (it = _atomspec_array.begin(); it != _atomspec_array.end(); ++it) {
        (*it)->computePower();
    }
    return;
}

void Spectrum::computePowerGradients() {
    for (auto it = _atomspec_array.begin(); it != _atomspec_array.end(); ++it) {
        (*it)->computePowerGradients();
    }
}

AtomicSpectrum *Spectrum::computeAtomic(Particle *center) {
    return this->computeAtomic(center, _structure->particles());
}

boost::python::object Spectrum::getDistanceMatrixNumpy(std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    Structure::laplace_t out(_atomspec_array.size(), _atomspec_array.size(), 0.0);
    this->getDistanceMatrix(out);
    return npc.ublas_to_numpy<Structure::dtype_t>(out);
}

void Spectrum::getDistanceMatrix(Structure::laplace_t &out) {
    assert(out.size1() == _atomspec_array.size() && out.size2() == _atomspec_array.size() && 
        "Output dimensions inconsistent with spectrum");
    int i = 0;
    for (auto it = _atomspec_array.begin(); it != _atomspec_array.end(); ++it, ++i) {
        int j = i;
        for (auto jt = it; jt != _atomspec_array.end(); ++jt, ++j) {
            vec dr = _structure->connect((*it)->getCenterPos(), (*jt)->getCenterPos());
            double r = soap::linalg::abs(dr);
            out(i,j) = r;
            out(j,i) = r;
        }
    }
}

AtomicSpectrum *Spectrum::computeAtomic(
        Particle *center, 
        Structure::particle_array_t &targets) {
    GLOG() << "Compute atomic spectrum for particle " << center->getId()
        << " (type " << center->getType() 
        << ", targets " << targets.size() 
        << ") ..." << std::endl;

    // FIND IMAGE REPITIONS REQUIRED TO SATISFY CUTOFF
    vec box_a = _structure->getBoundary()->getBox().getCol(0);
    vec box_b = _structure->getBoundary()->getBox().getCol(1);
    vec box_c = _structure->getBoundary()->getBox().getCol(2);
    double rc = _basis->getCutoff()->getCutoff();
    std::vector<int> na_nb_nc = _structure->getBoundary()->calculateRepetitions(rc);
    int na_max = na_nb_nc[0];
    int nb_max = na_nb_nc[1];
    int nc_max = na_nb_nc[2];

    // CREATE BLANK
    AtomicSpectrum *atomic_spectrum = new AtomicSpectrum(center, this->_basis);

    Structure::particle_it_t pit;
    for (pit = targets.begin(); pit != targets.end(); ++pit) {
        if (_options->doExcludeTarget((*pit)->getType()) ||
            _options->doExcludeTargetId((*pit)->getId())) continue;
        for (int na=-na_max; na<na_max+1; ++na) {
        for (int nb=-nb_max; nb<nb_max+1; ++nb) {
        for (int nc=-nc_max; nc<nc_max+1; ++nc) {
            vec L = na*box_a + nb*box_b + nc*box_c;
            // FIND DISTANCE & DIRECTION, APPLY CUTOFF (= WEIGHT REDUCTION)
            vec dr = _structure->connect(center->getPos(), (*pit)->getPos()) + L;
            double r = soap::linalg::abs(dr);
            if (! this->_basis->getCutoff()->isWithinCutoff(r)) continue;
            vec d = (r > 0.) ? dr/r : vec(0.,0.,1.);
            bool is_image = (*pit == center);
            bool is_center = (*pit == center && na==0 && nb==0 && nc==0);
            double weight0 = (*pit)->getWeight();
            double weight_scale = _basis->getCutoff()->calculateWeight(r);
            if (is_center) {
                weight0 *= _basis->getCutoff()->getCenterWeight();
            }
            GLOG() << (*pit)->getType() << " X " << dr.getX() << " Y " << dr.getY() << " Z " << dr.getZ() 
                << " W " << (*pit)->getWeight() << " W0 " << weight0 << " S " << (*pit)->getSigma() 
                << std::endl;
            // COMPUTE EXPANSION & ADD TO SPECTRUM
            bool gradients = (is_image) ? 
                false : _options->get<bool>("spectrum.gradients");
            // TODO BasisExpansion is managed by AtomicSpectrum, create there:
            BasisExpansion *nb_expansion = new BasisExpansion(this->_basis);
            nb_expansion->computeCoefficients(r, d, weight0, weight_scale, (*pit)->getSigma(), gradients);
            atomic_spectrum->addQnlmNeighbour(*pit, nb_expansion);
        }}}
    }

    return atomic_spectrum;
}

void Spectrum::compute2D() {
    assert(_structure->hasLaplacian());
    Structure::laplace_t &laplacian = _structure->getLaplacian();
    this->compute2D(
        _structure->particles(), 
        _structure->particles(),
        laplacian);
}

void Spectrum::compute2D(
        Structure::particle_array_t &centers, 
        Structure::particle_array_t &targets,
        Structure::laplace_t &laplacian) {
    GLOG() << "Compute 2D spectrum "
        << "(centers " << centers.size() << ", targets " << targets.size() << ") ..." << std::endl;
    GLOG() << _options->summarizeOptions() << std::endl;
    GLOG() << "Using radial basis of type '" << _basis->getRadBasis()->identify() << "'" << std::endl;
    GLOG() << "Using angular basis of type '" << _basis->getAngBasis()->identify() << "'" << std::endl;
    GLOG() << "Using cutoff function of type '" << _basis->getCutoff()->identify() << "'" << std::endl;

    Structure::particle_it_t pit;
    for (pit = centers.begin(); pit != centers.end(); ++pit) {
        // Continue if exclusion defined ...
        if (_options->doExcludeCenter((*pit)->getType()) ||
            _options->doExcludeCenterId((*pit)->getId())) continue;
        // Compute ...
        AtomicSpectrum *atomic_spectrum = this->computeAtomic2D(*pit, targets, laplacian);
        this->addAtomic(atomic_spectrum);
    }
}

AtomicSpectrum *Spectrum::computeAtomic2D(
        Particle *center, 
        Structure::particle_array_t &targets, 
        Structure::laplace_t &laplacian) {
    GLOG() << "Compute atomic 2D spectrum for particle " << center->getId()
        << " (type " << center->getType() << ", targets " << targets.size() << ") ..." << std::endl;

    // NOTE 2D => Minimum-image convention in line with specified Laplacian
    double rc = _basis->getCutoff()->getCutoff();

    // CREATE BLANK
    AtomicSpectrum *atomic_spectrum = new AtomicSpectrum(center, this->_basis);
    Structure::particle_it_t pit;
    for (pit = targets.begin(); pit != targets.end(); ++pit) {

        // CHECK FOR EXCLUSIONS
        if (_options->doExcludeTarget((*pit)->getType()) ||
            _options->doExcludeTargetId((*pit)->getId())) continue;

        // FIND DISTANCE & DIRECTION, CHECK CUTOFF
        vec dr = _structure->connect(center->getPos(), (*pit)->getPos());
        double r_euclidean = soap::linalg::abs(dr);
        double r_laplace = laplacian(center->getId()-1, (*pit)->getId()-1);
        if (! this->_basis->getCutoff()->isWithinCutoff(r_laplace)) continue;
        vec d = (r_euclidean > 0.) ? dr/r_euclidean : vec(0.,0.,1.);

        // APPLY CUTOFF (= WEIGHT REDUCTION)
        bool is_image = false;
        bool is_center = (*pit == center);
        double weight0 = (*pit)->getWeight();
        double weight_scale = _basis->getCutoff()->calculateWeight(r_laplace);
        if (is_center) {
            weight0 *= _basis->getCutoff()->getCenterWeight();
        }

        GLOG() << (*pit)->getType() << " X " << dr.getX() << " Y " << dr.getY() << " Z " << dr.getZ() 
            << " W " << (*pit)->getWeight() << " S " << (*pit)->getSigma() << " rl " << r_laplace << std::endl;

        // COMPUTE EXPANSION & ADD TO SPECTRUM
        bool gradients = (is_image) ? false : _options->get<bool>("spectrum.gradients");
        BasisExpansion *nb_expansion = new BasisExpansion(this->_basis); // <- kept by AtomicSpectrum
        nb_expansion->computeCoefficients(r_laplace, d, weight0, weight_scale, (*pit)->getSigma(), gradients);
        atomic_spectrum->addQnlmNeighbour(*pit, nb_expansion);

    } // Close loop over particles

    return atomic_spectrum;
}

AtomicSpectrum *Spectrum::computeGlobal() {
    if (_global_atomic) throw soap::base::APIError("<Spectrum::computeGlobal> Already initialised.");
    _global_atomic = new AtomicSpectrum(_basis);
    GLOG() << "Computing global spectrum ..." << std::endl;
    bool gradients = _options->get<bool>("spectrum.gradients");
    for (auto it = _atomspec_array.begin(); it != _atomspec_array.end(); ++it) {
        GLOG() << "  Adding center " << (*it)->getCenter()->getId()
            << " (type " << (*it)->getCenter()->getType() << ")" << std::endl;
        _global_atomic->mergeQnlm(*it, 0.5, gradients); // <- Scale factor 0.5 necessary in order not to overcount pairs
    }
    _global_atomic->computePower();
    if (gradients) _global_atomic->computePowerGradients();
    return _global_atomic;
}

AtomicSpectrum *Spectrum::getAtomic(int slot_idx, std::string center_type) {
	AtomicSpectrum *atomic_spectrum = NULL;
	if (center_type == "") {
		// NO TYPE => QUERY ATMSPEC ARRAY
		if (slot_idx < _atomspec_array.size()) {
			atomic_spectrum = _atomspec_array[slot_idx];
		}
		else {
			throw soap::base::OutOfRange("Spectrum slot index");
		}
	}
	else {
		// TYPE => QUERY ATMSPEC MAP
		map_atomspec_array_t::iterator it = _map_atomspec_array.find(center_type);
		if (it == _map_atomspec_array.end()) {
			throw soap::base::OutOfRange("No spectrum of type '" + center_type + "'");
		}
		else {
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

void Spectrum::writeDensityOnGrid(
        int slot_idx, std::string center_type, std::string density_type) {
	AtomicSpectrum *atomic_spectrum = this->getAtomic(slot_idx, center_type);
	// WRITE CUBE FILES
	if (atomic_spectrum) {
		atomic_spectrum->getQnlm(density_type)->writeDensityOnGrid(
			"density.explicit.cube", _options, _structure, atomic_spectrum->getCenter(), false);
	}
	return;
}

void Spectrum::writeDensityCubeFile(
        int atom_idx, std::string density_type, std::string filename, bool from_expansion) {
    if (atom_idx < _atomspec_array.size()) {
        AtomicSpectrum *atomic_spectrum = this->_atomspec_array[atom_idx];
		atomic_spectrum->getQnlm(density_type)->writeDensityOnGrid(
			filename, _options, _structure, atomic_spectrum->getCenter(), from_expansion);
    }
    else {
        throw soap::base::OutOfRange("Spectrum::writeDensityCubeFile");
    }
    return;
}

void Spectrum::writeDensityOnGridInverse(
        int slot_idx, std::string center_type, std::string type1, std::string type2) {
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

std::string Spectrum::saves(bool prune) {
    if (prune) {
        // Delete all pid-resolved data from atomic spectra.
        // Note that this data is required to compute gradients.
        // Pruning will no longer allow computing gradients
        // for serialised objects.
        for (auto it = _atomspec_array.begin(); it != _atomspec_array.end(); ++it) {
            (*it)->prunePidData();
        }
        if (_global_atomic) {
            _global_atomic->prunePidData();
        }
    }
    std::stringstream bstream;
    boost::archive::binary_oarchive arch(bstream);
    arch << (*this);
    return bstream.str();
}

void Spectrum::load(std::string archfile) {
	std::ifstream ifs(archfile.c_str());
	boost::archive::binary_iarchive arch(ifs);
	arch >> (*this);
	return;
}

Spectrum &Spectrum::loads(std::string bstr) {
    std::stringstream bstream;
    bstream << bstr;
    boost::archive::binary_iarchive arch(bstream);
    arch >> (*this);
    return (*this);
}

void Spectrum::registerPython() {
    using namespace boost::python;
    void (Spectrum::*computeAll)() = &Spectrum::compute;
    void (Spectrum::*computeSeg)(Segment*) = &Spectrum::compute;
    void (Spectrum::*computeSegPair)(Segment*, Segment*) = &Spectrum::compute;
    void (Spectrum::*computeCentersTargets)(Structure::particle_array_t&, 
        Structure::particle_array_t&) = &Spectrum::compute;

    class_<Spectrum>("Spectrum", init<Structure &, Options &>())
    	.def(init<Structure &, Options &, Basis &>())
    	.def(init<std::string>())
        .def(init<>())
    	.def("__iter__", 
            range<return_value_policy<reference_existing_object> >(
            &Spectrum::beginAtomic, &Spectrum::endAtomic))
        .def("__len__", &Spectrum::length)
	    .def("compute", computeAll)
        .def("compute", computeSeg)
	    .def("compute", computeSegPair)
	    .def("compute", computeCentersTargets)
		.def("computePower", &Spectrum::computePower)
		.def("computePowerGradients", &Spectrum::computePowerGradients)
        .def("deleteGlobal", &Spectrum::deleteGlobal)
		.def("computeGlobal", &Spectrum::computeGlobal, 
            return_value_policy<reference_existing_object>())
		.def("addAtomic", &Spectrum::addAtomic)
		.def("getAtomic", &Spectrum::getAtomic, 
            return_value_policy<reference_existing_object>())
		.def("getGlobal", &Spectrum::getGlobal, 
            return_value_policy<reference_existing_object>())
        .def("getDistanceMatrix", &Spectrum::getDistanceMatrixNumpy)
	    .def("saveAndClean", &Spectrum::saveAndClean)
		.def("save", &Spectrum::save)
		.def("load", &Spectrum::load)
        .def("saves", &Spectrum::saves)
        .def("loads", &Spectrum::loads, ref_existing())
        .def("writeDensityOnGrid", &Spectrum::writeDensityOnGrid)
        .def("writeDensityCubeFile", &Spectrum::writeDensityCubeFile)
		.def("writeDensity", &Spectrum::writeDensity)
		.def("writePowerDensity", &Spectrum::writePowerDensity)
		.def("writeDensityOnGridInverse", &Spectrum::writeDensityOnGridInverse)
		.add_property("options", make_function(&Spectrum::getOptions, ref_existing()))
		.add_property("basis", make_function(&Spectrum::getBasis, ref_existing()))
		.add_property("structure", make_function(&Spectrum::getStructure, ref_existing()));
    class_<atomspec_array_t>("AtomicSpectrumContainer")
           .def(vector_indexing_suite<atomspec_array_t>());
}

} // End soap namespace
