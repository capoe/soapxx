/*
Adapted from dscribe/ext/soapGTO.h:

Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  
#include <pybind11/stl.h>    
#include "celllist.hpp"
#include "soapgto.hpp"
#include "kernel.hpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(_soapgto, m) {
    m.def("evaluate_soapgto", &soapGTO, "SOAP with gaussian type orbital radial basis set");
    m.def("smooth_match", &smooth_match, "Smooth best-match assignment");
    py::class_<CellList>(m, "CellList")
        .def(py::init<py::array_t<double>, double>())
        .def("get_neighbours_for_index", &CellList::getNeighboursForIndex)
        .def("get_neighbours_for_position", &CellList::getNeighboursForPosition);
    py::class_<CellListResult>(m, "CellListResult")
        .def(py::init<>())
        .def_readonly("indices", &CellListResult::indices)
        .def_readonly("distances", &CellListResult::distances)
        .def_readonly("distances_squared", &CellListResult::distancesSquared);
}
