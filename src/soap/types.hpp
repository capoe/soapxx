#ifndef _SOAP_TYPES_H
#define	_SOAP_TYPES_H


#if PY_VERSION_HEX < 0x03000000
#include "Python.h"
#endif

#include <vector>
#include <string>
#include <stdexcept>
#include <boost/python.hpp>
#include <boost/lexical_cast.hpp>
#include "soap/linalg/types.hpp"
#include "soap/linalg/matrix.hpp"

namespace soap {

typedef soap::linalg::vec vec;
typedef soap::linalg::matrix matrix;

typedef boost::python::return_internal_reference<> ref_internal;
typedef boost::python::return_value_policy<boost::python::reference_existing_object> ref_existing;
typedef boost::python::return_value_policy<boost::python::copy_non_const_reference> copy_non_const;
typedef boost::python::return_value_policy<boost::python::copy_const_reference> copy_const;

template<typename target_t, typename source_t>
inline target_t lexical_cast(const source_t &arg, const std::string &error)
{
    try {
        return boost::lexical_cast<target_t,source_t>(arg);
    } catch(std::exception &err) {
        throw std::runtime_error("invalid type: " + error);
    }
}

}

#endif
