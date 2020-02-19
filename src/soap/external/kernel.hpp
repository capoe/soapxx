#ifndef KERNEL_EXT_HPP 
#define KERNEL_EXT_HPP 

#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

void smooth_match(
    py::array_t<double> P,
    py::array_t<double> K,
    int nx,
    int ny,
    double gamma,
    double epsilon,
    bool verbose);

#endif
