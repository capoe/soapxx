#include <boost/version.hpp>
#include "bindings.hpp"
#include "coulomb.hpp"
#include "fieldtensor.hpp"
#include "npfga.hpp"

namespace soap {

}

BOOST_PYTHON_MODULE(_soapxx)
{
    using namespace boost::python;

#if BOOST_VERSION >= 106400
    Py_Initialize();
    boost::python::numpy::initialize();
#endif
    soap::Structure::registerPython();
    soap::Segment::registerPython();
    soap::Particle::registerPython();
    soap::Options::registerPython();

    soap::Spectrum::registerPython();
    soap::Basis::registerPython();
    soap::AtomicSpectrum::registerPython();
    soap::BasisExpansion::registerPython();
    soap::PowerExpansion::registerPython();
    soap::Mol2D::registerPython();

    soap::EnergySpectrum::registerPython();
    soap::HierarchicalCoulomb::registerPython();
    soap::AtomicSpectrumHC::registerPython();
    soap::FTSpectrum::registerPython();
    soap::AtomicSpectrumFT::registerPython();

    soap::RadialBasisFactory::registerAll();
    soap::AngularBasisFactory::registerAll();
    soap::CutoffFunctionFactory::registerAll();

    soap::npfga::FNode::registerPython();
    soap::npfga::FGraph::registerPython();

    boost::python::def("silence", &soap::GLOG_SILENCE);
    boost::python::def("verbose", &soap::GLOG_VERBOSE);
}
