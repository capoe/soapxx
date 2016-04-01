#include "bindings.hpp"

namespace soap {

}

BOOST_PYTHON_MODULE(_soapxx)
{
    using namespace boost::python;
    soap::Particle::registerPython();
    soap::Segment::registerPython();
    soap::Structure::registerPython();
    soap::Options::registerPython();
    soap::Spectrum::registerPython();

    soap::RadialBasisFactory::registerAll();
    soap::AngularBasisFactory::registerAll();
    soap::CutoffFunctionFactory::registerAll();
}

