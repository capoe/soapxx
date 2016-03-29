#include "spectrum.hpp"

namespace soap {


Spectrum::Spectrum(Structure &structure, Options &options)
    : _structure(&structure), _options(&options), _log(NULL) {
	GLOG() << "Configuring spectrum ..." << std::endl;
	// CONFIGURE RADIAL BASIS
    _radbasis = RadialBasisOutlet().create(_options->get<std::string>("radialbasis.type"));
    _radbasis->configure(options);
    // CONFIGURE ANGULAR BASIS
    _angbasis = new AngularBasis();
    _angbasis->configure(options);
}

Spectrum::~Spectrum() {
	delete _log;
	_log = NULL;

	delete _radbasis;
	_radbasis = NULL;

	delete _angbasis;
	_angbasis = NULL;
}

void Spectrum::compute() {
	GLOG() << "Compute spectrum ..." << std::endl;
    GLOG() << _options->summarizeOptions() << std::endl;
    GLOG() << "Using radial basis of type '" << _radbasis->identify() << "'" << std::endl;
    GLOG() << "Using angular basis of type '" << _angbasis->identify() << "'" << std::endl;

    Structure::particle_it_t pit;
    for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
    	this->computeAtomic(*pit);
    	// TODO Receive & store atomic spectrum
    	break;
    }
}

void Spectrum::computeAtomic(Particle *center) {
	GLOG() << "Compute atomic spectrum for particle " << center->getId()
	    << " (type " << center->getType() << ")" << std::endl;
    Structure::particle_it_t pit;

    struct BasisCoefficients
	{
		BasisCoefficients(RadialCoefficients &c_n, AngularCoefficients &c_lm)
		: _c_n(c_n), _c_lm(c_lm) {
			_c_nlm.resize(c_n.size(), c_lm.size());
			_c_nlm = ub::outer_prod(_c_n, _c_lm);
		}
		std::complex<double> &get(int n, int l, int m) {
			if (_c_lm.checkSize(l, m) && _c_n.checkSize(n)) {
				return _c_nlm(n, l*l+l+m);
			}
			else {
				throw soap::base::OutOfRange("BasisCoefficients::get");
			}
		}
		void add(BasisCoefficients &other) {
			_c_nlm = _c_nlm + other._c_nlm;
		}

		ub::matrix< std::complex<double> > _c_nlm;
		RadialCoefficients _c_n;
		AngularCoefficients _c_lm;
	};



    RadialCoefficients c_n_zero = _radbasis->computeCoefficientsAllZero();
    AngularCoefficients c_lm_zero = _angbasis->computeCoefficientsAllZero();
    BasisCoefficients c_nlm(c_n_zero, c_lm_zero);

    std::string type_this = center->getType();
    for (pit = _structure->beginParticles(); pit != _structure->endParticles(); ++pit) {
    	vec dr = _structure->connect(center->getPos(), (*pit)->getPos());
    	double r = soap::linalg::abs(dr);
    	// TODO Cut-off check
    	vec d = dr/r;
    	std::string type_other = (*pit)->getType();

        // COMPUTE RADIAL COEFFICIENTS
    	RadialCoefficients c_n_pair = _radbasis->computeCoefficients(r);
    	/*
    	GLOG() << "Radial coefficients r = " << r << std::endl;
        for (int i = 0; i < c_n.size(); ++i) {
            GLOG() << c_n[i] << " | ";
        }
        GLOG() << std::endl;
        */

        // COMPUTE ANGULAR COEFFICIENTS
        AngularCoefficients c_lm_pair = _angbasis->computeCoefficients(d, r);
        /*
    	GLOG() << "Angular coefficients d = " << d << std::endl;
		for (int lm = 0; lm < c_lm.size(); ++lm) {
			GLOG() << c_lm[lm] << " | ";
		}
		GLOG() << std::endl;
		*/

        BasisCoefficients c_nlm_pair(c_n_pair, c_lm_pair);
        c_nlm.add(c_nlm_pair);

    }

    // SAMPLE EXPANDED DENSITY OVER GRID
    int I = 20;
    double dx = 0.1;

    std::cout << "Density on grid" << std::endl;
    std::cout << "Loop order X -> Y -> Z" << std::endl;
    std::cout << "1 -2 -2 -2" << std::endl;
    std::cout << "41 0.1 0.0 0.0" << std::endl;
    std::cout << "41 0.0 0.1 0.0" << std::endl;
    std::cout << "41 0.0 0.0 0.1" << std::endl;
    std::cout << "6  0.0 0.0 0.0 0.0" << std::endl;

    for (int i = -I; i <= I; ++i) {
    	//std::cout << i << std::endl;
    	for (int j = -I; j <= I; ++j) {
    		for (int k = -I; k <= I; ++k) {


    			vec dr(i*dx, j*dx, k*dx);
    			double r = soap::linalg::abs(dr);
    			vec d = dr/r;
    			RadialCoefficients c_n_dr = _radbasis->computeCoefficients(r);
    			AngularCoefficients c_lm_dr = _angbasis->computeCoefficients(d, r);
    			c_lm_dr.conjugate();
    			BasisCoefficients c_nlm_dr(c_n_dr, c_lm_dr);

    			//double density_dr = ub::inner_prod(c_nlm, c_nlm_dr);
    			std::complex<double> density_dr(0.,0.);
    			for (int n = 0; n < _radbasis->N(); ++n) {
					for (int l = 0; l <= _angbasis->L(); ++l) {
						for (int m = -l; m <= l; ++m) {
							std::complex<double> Gn_Ylm = c_nlm.get(n, l, m);
							density_dr += c_nlm.get(n, l, m)*c_nlm_dr.get(n, l, m);
						}
					}
				}

    			std::cout << density_dr.real() << " " << std::flush;
    			if ( ((k+I) % 6) == 5 ) {
    				std::cout << std::endl;
    			}
    		}
    	}
    }

    for (int n = 0; n < _radbasis->N(); ++n) {
    	for (int l = 0; l <= _angbasis->L(); ++l) {
    		for (int m = -l; m <= l; ++m) {
                std::complex<double> Gn_Ylm = c_nlm.get(n, l, m);

    		}
    	}
    }

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

