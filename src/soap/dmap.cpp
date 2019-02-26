#include <algorithm>
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
    dtype_t res = 0.0;
    dtype_t r12 = 0.0;
    // NOTE This is clear but inefficient, see improved version below.
    //for (auto it=dmap.begin(); it!=dmap.end(); ++it) {
    //    auto jt = other->dmap.find(it->first);
    //    if (jt != other->end()) {
    //        // Manual version, for speed-up see library version below
    //        //auto &c1 = *(it->second);
    //        //auto &c2 = *(jt->second);
    //        //res += c1.dot(c2);
    //        //res += ub::inner_prod(c1, c2);
    //        //for (int i=0; i<c1.size(); ++i) {
    //        //    res += c1(i)*c2(i);
    //        //}
    //        soap::linalg::linalg_dot(*(it->second), *(jt->second), r12);
    //        res += r12;
    //    }
    //}
    auto it=this->begin();
    auto jt=other->begin();
    while (it != this->end()) {
        while (jt != other->end() && jt->first < it->first) ++jt;
        if (jt == other->end()) break;
        if (it->first == jt->first) {
            soap::linalg::linalg_dot(*(it->second), *(jt->second), r12);
            res += r12;
        }
        ++it;
    }
    return double(res);
}

void DMap::add(DMap *other) {
    dmap_t add_entries; 
    auto it = this->begin();
    auto jt = other->begin();
    while (jt != other->end()) {
        if (it == this->end() || jt->first < it->first) {
            vec_t *add_vec = new vec_t(jt->second->size());
            *add_vec = *(jt->second);
            add_entries.push_back(channel_t(jt->first, add_vec));
            ++jt;
        } else if (jt->first == it->first) {
            *(it->second) += *(jt->second);
            ++it;
            ++jt;
        } else if (jt->first > it->first) {
            ++it;
        }
    }
    for (auto it=add_entries.begin(); it!=add_entries.end(); ++it) {
        dmap.push_back(*it);
    }
    this->sort();
}

void DMap::sort() {
    std::sort(dmap.begin(), dmap.end(), 
        [](DMap::channel_t c1, DMap::channel_t c2) {
            return c1.first <= c2.first;
        }
    );
}

boost::python::list DMap::listChannels() {
    boost::python::list channel_keys;
    for (auto it=begin(); it!=end(); ++it) {
        std::string c12 = ENCODER.decode(it->first);
        channel_keys.append(c12);
    }
    return channel_keys;
}

double DMap::dotFilter(DMap *other) {
    if (other->filter != this->filter) return 0.0;
    else return this->dot(other);
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
        // Exclude symmetric redundancies (e.g., project C:H, but not H:C
        if (it->first.first > it->first.second) continue; 
        auto coeff = it->second->getCoefficients();
        int length = coeff.size1()*coeff.size2();
        auto v = new vec_t(length);
        int c = 0;
        for (int i=0; i<coeff.size1(); ++i) {
            for (int j=0; j<coeff.size2(); ++j, ++c) {
                (*v)(c) = coeff(i,j).real();
            }
        }
        //dmap[it->first] = v;
        TypeEncoder::code_t e = ENCODER.encode(it->first.first, it->first.second);
        channel_t p(e, v);
        dmap.push_back(p);
    }
    this->sort();
    double norm = std::sqrt(this->dot(this));
    this->multiply(1./norm);
    filter = atomic->getCenterType();
}

void DMap::registerPython() {
    using namespace boost::python;
    class_<DMap, DMap*>("DMap", init<>())
        .add_property("filter", &DMap::getFilter)
        .def("listChannels", &DMap::listChannels)
        .def("dot", &DMap::dot)
        .def("dotFilter", &DMap::dotFilter);
}

DMapMatrix::DMapMatrix() : is_view(false) {
    ;
}

DMapMatrix::DMapMatrix(std::string archfile) : is_view(false) {
    this->load(archfile);
}

DMapMatrix::DMapMatrix(bool set_as_view) : is_view(set_as_view) {
    ;
}

DMapMatrix::~DMapMatrix() {
    this->clear();
}

void DMapMatrix::clear() {
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

void DMapMatrix::sum() {
    auto *summed = new DMap();
    summed->filter = "";
    for (auto it=begin(); it!=end(); ++it) {
        summed->add(*it);
    }
    this->clear();
    dmm.push_back(summed); 
}

void DMapMatrix::dot(DMapMatrix *other, matrix_t &output) {
    assert(output.size1() == this->rows() && output.size2() == other->rows() &&
        "Output matrix dimensions incompatible with input"); 
    int i = 0;
    for (auto it=begin(); it!=end(); ++it, ++i) {
        int j = 0;
        for (auto jt=other->begin(); jt!=other->end(); ++jt, ++j) {
            output(i,j) = (*it)->dot(*jt);
        }
    }
}

void DMapMatrix::dotFilter(DMapMatrix *other, matrix_t &output) {
    assert(output.size1() == this->rows() && output.size2() == other->rows() &&
        "Output matrix dimensions incompatible with input"); 
    int i = 0;
    for (auto it=begin(); it!=end(); ++it, ++i) {
        int j = 0;
        for (auto jt=other->begin(); jt!=other->end(); ++jt, ++j) {
            output(i,j) = (*it)->dotFilter(*jt);
        }
    }
}

void dmm_inner_product(DMapMatrix &AX, DMapMatrix &BX, double power, 
        bool filter, DMapMatrix::matrix_t &output) {
    assert(output.size1() == AX.rows() && output.size2() == BX.rows() &&
        "Output matrix dimensions incompatible with input"); 
    if (filter) {
        int i = 0;
        for (auto it=AX.begin(); it!=AX.end(); ++it, ++i) {
            int j = 0;
            for (auto jt=BX.begin(); jt!=BX.end(); ++jt, ++j) {
                output(i,j) = std::pow((*it)->dotFilter(*jt), power);
            }
        }
    } else {
        int i = 0;
        for (auto it=AX.begin(); it!=AX.end(); ++it, ++i) {
            int j = 0;
            for (auto jt=BX.begin(); jt!=BX.end(); ++jt, ++j) {
                output(i,j) = std::pow((*it)->dot(*jt), power);
            }
        }
    }
}

boost::python::object DMapMatrix::dotNumpy(DMapMatrix *other, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    DMapMatrix::matrix_t output(this->rows(), other->rows(), 0.0);
    this->dot(other, output);
    return npc.ublas_to_numpy<double>(output);
}

boost::python::object DMapMatrix::dotFilterNumpy(DMapMatrix *other, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    DMapMatrix::matrix_t output(this->rows(), other->rows(), 0.0);
    this->dotFilter(other, output);
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
        .def(init<std::string>())
        .add_property("rows", &DMapMatrix::rows)
        .def("__len__", &DMapMatrix::rows)
        .def("__getitem__", &DMapMatrix::getRow, return_value_policy<reference_existing_object>())
        .def("addView", &DMapMatrix::addView)
        .def("getView", &DMapMatrix::getView, return_value_policy<reference_existing_object>())
        .def("append", &DMapMatrix::append)
        .def("sum", &DMapMatrix::sum)
        .def("dot", &DMapMatrix::dotNumpy)
        .def("dotFilter", &DMapMatrix::dotFilterNumpy)
        .def("load", &DMapMatrix::load)
        .def("save", &DMapMatrix::save);
}

DMapMatrixSet::DMapMatrixSet() {
    ;
}

DMapMatrixSet::DMapMatrixSet(std::string archfile) {
    this->load(archfile);
}

DMapMatrixSet::~DMapMatrixSet() {
    for (auto it=begin(); it!=end(); ++it) delete *it;
    dset.clear();
}

void DMapMatrixSet::append(DMapMatrix *dmap) {
    dset.push_back(dmap);
}

void DMapMatrixSet::save(std::string archfile) {
    std::ofstream ofs(archfile.c_str());
    boost::archive::binary_oarchive arch(ofs);
    arch << (*this);
    return;
}

void DMapMatrixSet::load(std::string archfile) {
	std::ifstream ifs(archfile.c_str());
	boost::archive::binary_iarchive arch(ifs);
	arch >> (*this);
	return;
}

void DMapMatrixSet::registerPython() {
    using namespace boost::python;
    class_<DMapMatrixSet, DMapMatrixSet*>("DMapMatrixSet", init<>())
        .def(init<std::string>())
        .def("__len__", &DMapMatrixSet::size)
        .def("__getitem__", &DMapMatrixSet::get, return_value_policy<reference_existing_object>())
        .add_property("size", &DMapMatrixSet::size)
        .def("save", &DMapMatrixSet::save)
        .def("load", &DMapMatrixSet::load)
        .def("append", &DMapMatrixSet::append);
}

BlockLaplacian::BlockLaplacian() : n_rows(0), n_cols(0) {
    ;
}

BlockLaplacian::BlockLaplacian(std::string archfile) : n_rows(0), n_cols(0) {
    this->load(archfile);
}

BlockLaplacian::~BlockLaplacian() {
    for (auto it=begin(); it!=end(); ++it) {
        delete *it;
    }
    blocks.clear();
}

BlockLaplacian::block_t *BlockLaplacian::addBlock(int n_rows_block, int n_cols_block) {
    block_t *new_block = new block_t(n_rows_block, n_cols_block);
    n_rows += n_rows_block;
    n_cols += n_cols_block;
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

boost::python::object BlockLaplacian::dotNumpy(
        boost::python::object &np_other, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    block_t other;
    npc.numpy_to_ublas<dtype_t>(np_other, other);
    block_t output(this->rows(), other.size2());
    this->dot(other, output);
    return npc.ublas_to_numpy<double>(output);
}

void BlockLaplacian::dot(block_t &other, block_t &output) {
    assert(n_cols == other.size1() && output.size1() == n_rows 
        && output.size2() == other.size2() && "Inconsistent matrix dimensions");
    int i_off = 0;
    int j_off = 0;
    for (auto it=begin(); it!=end(); ++it) {
        block_t &block = *(*it);
        // Manual version, for speed-up see library version below
        //for (int i=0; i<block.size1(); ++i) {
        //    for (int k=0; k<other.size2(); ++k) {
        //        double out_ik = 0.0;
        //        for (int j=0; j<block.size2(); ++j) {
        //            out_ik += block(i,j)*other(j_off+j, k);
        //        }
        //        output(i_off+i,k) = out_ik;
        //    }
        //}
        soap::linalg::linalg_matrix_block_dot(
            block,
            other,
            output,
            i_off,
            j_off);
        i_off += block.size1();
        j_off += block.size2();
    }
}

void BlockLaplacian::registerPython() {
    using namespace boost::python;
    class_<BlockLaplacian, BlockLaplacian*>("BlockLaplacian", init<>())
        .def(init<std::string>())
        .add_property("rows", &BlockLaplacian::rows)
        .add_property("cols", &BlockLaplacian::cols)
        .def("dot", &BlockLaplacian::dotNumpy)
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

void Proto::parametrize(DMapMatrix &AX_in, DMapMatrix &BX_in, BlockLaplacian &DAB) {
    GLOG() << "Build Proto from AX x BX = " << AX_in.rows() << " x " << BX_in.rows() << std::endl;
    AX = &AX_in;
    BX = &BX_in;
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
                // Apply cutoff and Jacobian (here: 1/r^2)
                gab_block(i,j) = cutoff->calculateWeight(dab_block(i,j)) / pow(dab_block(i,j),2);
            }
        }
    }
    GLOG() << std::endl;
    Gnab.push_back(new_gab);
}

boost::python::object Proto::projectPython(DMapMatrix *ax, DMapMatrix *bx, 
        double xi, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    DMapMatrix::matrix_t output(ax->rows(), bx->rows(), 0);
    this->project(ax, bx, xi, output);
    return npc.ublas_to_numpy<double>(output);
}

void Proto::project(DMapMatrix *ax, DMapMatrix *bx, double xi, matrix_t &output) {
    // NOTE The key constraint here is to avoid matrices of size NA x NB
    bool filter = true; // <- = true means that kab = 0 if element a != element b
    int NA = AX->rows();
    int NB = BX->rows();
    int na = ax->rows();
    int nb = bx->rows();
    assert(output.size1() == na && output.size2() == nb
        && "Inconsistent output matrix dimensions");
    // Project environments
    GLOG() << "Allocating " << na << "x" << NA << " and " 
        << NB << "x" << nb << " kernel matrices" << std::endl;
    DMapMatrix::matrix_t K_aA(na, NA);
    DMapMatrix::matrix_t K_Bb(NB, nb);
    GLOG() << "Environment projection KaA=ax.AX" << std::endl;
    dmm_inner_product(*ax, *AX, xi, filter, K_aA);
    GLOG() << "Environment projection KBb=BX.bx" << std::endl;
    dmm_inner_product(*BX, *bx, xi, filter, K_Bb);
    // Project basis
    GLOG() << "Basis projection HAb=GAB.KBb" << std::endl;
    DMapMatrix::matrix_t H_Ab(NA, nb);
    assert(Gnab.size() == 1);
    GLOG() << "Basis projection Eab=KaA.HAb" << std::endl;
    Gnab[0]->dot(K_Bb, H_Ab); // TODO Account for more than one basis fct
    GLOG() << "Projecting KaA.HAb" << std::endl;
    soap::linalg::linalg_matrix_dot(K_aA, H_Ab, output);
    // Normalization
    GLOG() << "Normalization of Eab" << std::endl;
    std::vector<double> za(na);
    std::vector<double> zb(nb);
    for (int i=0; i<na; ++i) {
        for (int j=0; j<NA; ++j) {
            za[i] += K_aA(i,j);
        }
    }
    for (int i=0; i<nb; ++i) {
        for (int j=0; j<NB; ++j) {
            zb[i] += K_Bb(j,i);
        }
    }
    for (int i=0; i<na; ++i) {
        for (int j=0; j<nb; ++j) {
            output(i,j) = - std::log(output(i,j)/(za[i]*zb[j]));
        }
    }
}

void Proto::registerPython() {
    using namespace boost::python;
    class_<Proto, Proto*>("Proto", init<>())
        .def("project", &Proto::projectPython)
        .def("parametrize", &Proto::parametrize);
}

TypeEncoder::TypeEncoder() {
    encoder = encoder_t {
      { "H" ,   0 },
      { "C" ,   1 },
      { "N" ,   2 },
      { "O" ,   3 },
      { "F" ,   4 },
      { "P" ,   5 },
      { "S" ,   6 },
      { "Cl",   7 },
      { "Br",   8 },
      { "I" ,   9 }
    };
}

TypeEncoder::~TypeEncoder() {
    ;
}

void TypeEncoder::clear() {
    encoder.clear();
    order.clear();
}

void TypeEncoder::add(std::string type) {
    auto it = encoder.find(type);
    if (it != end()) {
        throw soap::base::SanityCheckFailed("Type already added: '"+type+"'");
    }
    encoder[type] = code_t(size());
    order.push_back(type);
}

TypeEncoder::code_t TypeEncoder::encode(std::string type) {
    auto it = encoder.find(type);
    if (it == end()) throw soap::base::OutOfRange("Encoder type '"+type+"'");
    return it->second;
}

TypeEncoder::code_t TypeEncoder::encode(std::string type1, std::string type2) {
    auto it = encoder.find(type1);
    auto jt = encoder.find(type2);
    if (it == end() || jt == end()) throw soap::base::OutOfRange(type1+":"+type2);
    return encoder[type1]*size() + encoder[type2];
}

std::string TypeEncoder::decode(TypeEncoder::code_t code) {
    code_t c2 = code % size();
    code_t c1 = (code-c2)/size();
    return order[c1]+":"+order[c2];
}

void TypeEncoder::list() {
    for (auto it=order.begin(); it!=order.end(); ++it) {
        GLOG() << *it << " : " << encoder[*it] << std::endl;
    }
}

boost::python::list TypeEncoder::getTypes() {
    boost::python::list type_list;
    for (auto t: order) type_list.append(t);
    return type_list;
}

TypeEncoder ENCODER;

void TypeEncoderUI::clear() {
    ENCODER.clear();
}

void TypeEncoderUI::list() {
    ENCODER.list();
}

void TypeEncoderUI::add(std::string type) {
    ENCODER.add(type);
}

boost::python::list TypeEncoderUI::types() {
    return ENCODER.getTypes();
}

TypeEncoder::code_t TypeEncoderUI::encode(std::string t1, std::string t2) {
    return ENCODER.encode(t1, t2);
}

void TypeEncoder::registerPython() {
    code_t (TypeEncoder::*encodeSingle)(std::string) 
        = &TypeEncoder::encode;
    code_t (TypeEncoder::*encodePair)(std::string, std::string) 
        = &TypeEncoder::encode;

    using namespace boost::python;
    class_<TypeEncoder, TypeEncoder*>("TypeEncoder", init<>())
        .def("clear", &TypeEncoder::clear)
        .def("add", &TypeEncoder::add)
        .def("encode", encodeSingle)
        .def("encode", encodePair);

    // Python use: soap.encoder.clear(); soap.encoder.add("U"); ...
    class_<TypeEncoderUI>("encoder", init<>())
        .def("clear", &TypeEncoderUI::clear)
        .staticmethod("clear")
        .def("add", &TypeEncoderUI::add)
        .staticmethod("add")
        .def("encode", &TypeEncoderUI::encode)
        .staticmethod("encode")
        .def("types", &TypeEncoderUI::types)
        .staticmethod("types")
        .def("list", &TypeEncoderUI::list)
        .staticmethod("list");
}

}
