#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "soap/dmap.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"
//#include "soap/linalg/Eigen/Dense"

namespace soap {

DMap::DMap() {
    ;
}

DMap::~DMap() {
    for (auto it=dmap.begin(); it!=dmap.end(); ++it) {
        delete it->second;
    }
    dmap.clear();
}

DMapMatrix::DMapMatrix() {
    ;
}

double DMap::dot(DMap *other) {
    if (other->size() < this->size()) return other->dot(this);
    double res = 0.0;
    for (auto it=dmap.begin(); it!=dmap.end(); ++it) {
        auto jt = other->dmap.find(it->first);
        if (jt != other->end()) {
            auto &c1 = *(it->second);
            auto &c2 = *(jt->second);
            double r12 = 0.0;
            soap::linalg::linalg_dot(c1, c2, r12);
            res += r12;
            //res += c1.dot(c2);
            //res += ub::inner_prod(c1, c2);
            //for (int i=0; i<c1.size(); ++i) {
            //    res += c1(i)*c2(i);
            //}
        }
    }
    return res;
}

void DMap::multiply(double c) {
    for (auto it=begin(); it!=end(); ++it) {
        auto &v = *(it->second);
        v *= c;
    }
}

void DMap::adapt(AtomicSpectrum *atomic) {
    auto spec = atomic->getXnklMap();
    for (auto it=spec.begin(); it!=spec.end(); ++it) {
        if (it->first.first > it->first.second) continue; // E.g., only C:H, not H:C
        auto coeff = it->second->getCoefficients();
        int length = coeff.size1()*coeff.size2();
        auto v = new vec_t(length);
        int c = 0;
        for (int i=0; i<coeff.size1(); ++i) {
            for (int j=0; j<coeff.size2(); ++j, ++c) {
                (*v)(c) = coeff(i,j).real();
            }
        }
        dmap[it->first] = v;
    }
    double norm = std::sqrt(this->dot(this));
    this->multiply(1./norm);
}

void DMap::registerPython() {
    using namespace boost::python;
    class_<DMap, DMap*>("DMap", init<>());
}

DMapMatrix::~DMapMatrix() {
    for (auto it=dmm.begin(); it!=dmm.end(); ++it) {
        delete (*it);
    }
    dmm.clear();
}

void DMapMatrix::append(Spectrum *spectrum) {
    for (auto it=spectrum->beginAtomic(); it!=spectrum->endAtomic(); ++it) {
        DMap *new_dmap = new DMap();
        new_dmap->adapt(*it);
        dmm.push_back(new_dmap);
    }
    for (auto it=dmm.begin(); it!=dmm.end(); ++it) {
        for (auto jt=(*it)->begin(); jt!=(*it)->end(); ++jt) {
            auto dm = jt->second;
        }
    }
}

void DMapMatrix::dot(DMapMatrix *other, ub_matrix_t &output) {
    assert(output.size1() == this->size() && output.size2() == other->size() &&
        "Output matrix dimensions incompatible with input"); 
    int i = 0;
    for (auto it=begin(); it!=end(); ++it, ++i) {
        int j = 0;
        for (auto jt=other->begin(); jt!=other->end(); ++jt, ++j) {
            output(i,j) = (*it)->dot(*jt);
        }
    }
}

boost::python::object DMapMatrix::dotNumpy(DMapMatrix *other, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    ub_matrix_t output(this->size(), other->size());
    this->dot(other, output);
    return npc.ublas_to_numpy<double>(output);
}

void DMapMatrix::save(std::string archfile) {
    std::ofstream ofs(archfile.c_str());
    boost::archive::binary_oarchive arch(ofs);
    arch << (*this);
    return;
}

void DMapMatrix::load(std::string archfile) {
	std::ifstream ifs(archfile.c_str());
	boost::archive::binary_iarchive arch(ifs);
	arch >> (*this);
	return;
}

void DMapMatrix::registerPython() {
    using namespace boost::python;
    class_<DMapMatrix, DMapMatrix*>("DMapMatrix", init<>())
        .def("__len__", &DMapMatrix::size)
        .def("append", &DMapMatrix::append)
        .def("dot", &DMapMatrix::dotNumpy)
        .def("load", &DMapMatrix::load)
        .def("save", &DMapMatrix::save);
}

BlockLaplacian::BlockLaplacian() : n_rows(0), n_cols(0) {
    ;
}

BlockLaplacian::~BlockLaplacian() {
    for (auto it=begin(); it!=end(); ++it) {
        delete *it;
    }
    blocks.clear();
}

void BlockLaplacian::appendNumpy(boost::python::object &np_array, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    block_t *new_block = new block_t();
    npc.numpy_to_ublas<dtype_t>(np_array, *new_block);
    n_rows += new_block->size1();
    n_cols += new_block->size2();
    blocks.push_back(new_block);
}

void BlockLaplacian::save(std::string archfile) {
    std::ofstream ofs(archfile.c_str());
    boost::archive::binary_oarchive arch(ofs);
    arch << (*this);
    return;
}

void BlockLaplacian::load(std::string archfile) {
	std::ifstream ifs(archfile.c_str());
	boost::archive::binary_iarchive arch(ifs);
	arch >> (*this);
	return;
}

void BlockLaplacian::registerPython() {
    using namespace boost::python;
    class_<BlockLaplacian, BlockLaplacian*>("BlockLaplacian", init<>())
        .add_property("rows", &BlockLaplacian::rows)
        .add_property("cols", &BlockLaplacian::cols)
        .def("append", &BlockLaplacian::appendNumpy)
        .def("save", &BlockLaplacian::save)
        .def("load", &BlockLaplacian::load);
}

}
