#ifndef GYLM_EXT_HPP 
#define GYLM_EXT_HPP 

#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

void _py_evaluate_xtunkl(
    py::array_t<double> _xitunkl,
    py::array_t<double> _qitnlm,
    int n_src,
    int n_types,
    int nmax,
    int lmax);

void evaluate_gylm(
    py::array_t<double> coeffs, 
    py::array_t<double> tgt_pos, 
    py::array_t<double> src_pos, 
    py::array_t<double> gnl_centres, 
    py::array_t<double> gnl_alphas, 
    py::array_t<int> tgt_types, 
    py::array_t<int> all_types, 
    double r_cut, 
    double r_cut_width, 
    int n_src, 
    int n_tgt, 
    int n_types, 
    int nmax, 
    int lmax,
    double part_sigma,
    bool wconstant,
    double wscale,
    double wcentre,
    double ldamp,
    bool power,
    bool verbose);

#endif
