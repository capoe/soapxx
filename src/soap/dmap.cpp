#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "soap/dmap.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"
//#include "soap/linalg/Eigen/Dense"

#include "soap/options.hpp"

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
    filter = atomic->getCenterType();
}

void DMap::registerPython() {
    using namespace boost::python;
    class_<DMap, DMap*>("DMap", init<>());
}

DMapMatrix::DMapMatrix() : is_view(false) {
    ;
}

DMapMatrix::DMapMatrix(bool set_as_view) : is_view(set_as_view) {
    ;
}

DMapMatrix::~DMapMatrix() {
    if (!is_view) {
        for (auto it=dmm.begin(); it!=dmm.end(); ++it) delete (*it);
    }
    dmm.clear();
    for (auto it=views.begin(); it!=views.end(); ++it) delete it->second;
    views.clear();
}

void DMapMatrix::addView(std::string filter) {
    DMapMatrix *view = new DMapMatrix(true);
    for (auto it=begin(); it!=end(); ++it) {
        if ((*it)->filter == filter) {
            view->dmm.push_back(*it);
        }
    }
    auto it = views.find(filter);
    if (it != views.end()) delete it->second;
    views[filter] = view;
}

DMapMatrix *DMapMatrix::getView(std::string filter) {
    auto it = views.find(filter);
    if (it == views.end())
        throw soap::base::OutOfRange("View with filter="+filter);
    return it->second;
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
        .def("addView", &DMapMatrix::addView)
        .def("getView", &DMapMatrix::getView, return_value_policy<reference_existing_object>())
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

BlockLaplacian::block_t *BlockLaplacian::addBlock(int n_rows, int n_cols) {
    block_t *new_block = new block_t(n_rows, n_cols);
    blocks.push_back(new_block);
    return new_block;
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

Proto::Proto() {
    Options options;
    options.set("radialcutoff.Rc", 3.75);
    options.set("radialcutoff.Rc_width", 0.5);
    options.set("radialcutoff.center_weight", 1.0);
    cutoff = CutoffFunctionOutlet().create("shifted-cosine");
    cutoff->configure(options);
}

Proto::~Proto() {
    if (cutoff) delete cutoff;
    cutoff = NULL;
    for (auto it=Gnab.begin(); it!=Gnab.end(); ++it) delete *it;
    Gnab.clear();
}

void Proto::parametrize(DMapMatrix &AX, DMapMatrix &BX, BlockLaplacian &DAB) {
    GLOG() << "Build Proto from AX x BX = " << AX.size() << " x " << BX.size() << std::endl;
    AXM = &AX;
    BXM = &BX;
    // TODO Formulate in terms of basis expansion with dedicated objects
    // such as in Gnab = basis->evaluate(DAB).
    // Example here: Flat radial basis with cutoff
    GLOG() << "Expanding pair distances" << std::endl;
    for (auto it=Gnab.begin(); it!=Gnab.end(); ++it) delete *it;
    Gnab.clear();
    int n_basis_fcts = 1;
    auto *new_gab = new BlockLaplacian();
    int i_block = 0;
    for (auto it=DAB.begin(); it!= DAB.end(); ++it, ++i_block) {
        GLOG() << "\r" << " - Block " << i_block << std::flush;
        auto &dab_block = *(*it);
        auto &gab_block = *(new_gab->addBlock(dab_block.size1(), dab_block.size2()));
        for (int i=0; i<dab_block.size1(); ++i) {
            for (int j=0; j<dab_block.size2(); ++j) {
                gab_block(i,j) = cutoff->calculateWeight(dab_block(i,j));
            }
        }
    }
    GLOG() << std::endl;
    Gnab.push_back(new_gab);
}

boost::python::object Proto::projectPython(DMapMatrix &AX, DMapMatrix &BX, double xi, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    matrix_t output(AX.size(), BX.size());
    this->project(AX, BX, xi, output);
    return npc.ublas_to_numpy<double>(output);
}

void Proto::project(DMapMatrix &AX, DMapMatrix &BX, double xi, matrix_t &output) {
    //K_aA = AX.dot(*AXM);
    //K_Bb = BXM->dot(BX);
    //GK_Ab = Gnab[0]->dotRight(KBb);
    //output = K_aA.dot(GK_Ab);
}

void Proto::registerPython() {
    using namespace boost::python;
    class_<Proto, Proto*>("Proto", init<>())
        .def("project", &Proto::projectPython)
        .def("parametrize", &Proto::parametrize);
}

}
