#ifndef YLM_EXT_HPP 
#define YLM_EXT_HPP 

#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

void _py_ylm(
        py::array_t<double> x, 
        py::array_t<double> y, 
        py::array_t<double> z,
        int n_pts,
        int lmax, 
        py::array_t<double> py_ylm_out);

void evaluate_ylm(double *x, double *y, double *z, double *r,
    int n_pts, int lmax, double *ylm_out);

#endif
