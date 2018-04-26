#define BOOST_PYTHON_STATIC_LIB  
#define BOOST_LIB_NAME "boost_numpy"
#include <boost/config/auto_link.hpp>
#include "soap/linalg/bindings.hpp"
#include "soap/linalg/kernel.hpp"

namespace soap { namespace linalg {


}}

BOOST_PYTHON_MODULE(_linalg) {
    using namespace boost::python;
    //boost::python::numpy::ndarray::set_module_and_type("numpy", "ndarray");
    //using namespace boost::python::numpy;
    soap::linalg::vec::registerPython();
    soap::linalg::matrix::registerPython();
    soap::linalg::KernelModule::registerPython();
}

