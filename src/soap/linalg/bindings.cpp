#include <boost/version.hpp>
#if BOOST_VERSION >= 106400 
#define BOOST_PYTHON_STATIC_LIB  
#define BOOST_LIB_NAME "boost_numpy"
#include <boost/config/auto_link.hpp>
#endif
#include "soap/linalg/bindings.hpp"
#include "soap/linalg/kernel.hpp"

namespace soap { namespace linalg {


}}

BOOST_PYTHON_MODULE(_linalg) {
    using namespace boost::python;
#if BOOST_VERSION < 106400
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
#endif
    soap::linalg::vec::registerPython();
    soap::linalg::matrix::registerPython();
    soap::linalg::KernelModule::registerPython();
}

