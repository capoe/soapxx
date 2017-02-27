#include "soap/linalg/bindings.hpp"
#include "soap/linalg/kernel.hpp"

namespace soap { namespace linalg {


}}

BOOST_PYTHON_MODULE(_linalg) {
    using namespace boost::python;
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    soap::linalg::vec::registerPython();
    soap::linalg::matrix::registerPython();
    soap::linalg::KernelModule::registerPython();
}

