#include <algorithm>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <sstream>

#include "soap/dmap.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"
//#include "soap/linalg/Eigen/Dense"

#include "soap/options.hpp"

namespace soap {

DMap::DMap() {
    ;
}

DMap::DMap(std::string filter_type) : filter(filter_type) {
    ;
}

DMap::~DMap() {
    for (auto it=pid_gradmap.begin(); it!=pid_gradmap.end(); ++it) {
        delete *it;
    }
    pid_gradmap.clear();
    for (auto it=dmap.begin(); it!=dmap.end(); ++it) {
        delete (*it).second;
    }
    dmap.clear();
}

bpy::object DMap::val(int chidx, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    return npc.ublas_to_numpy<dtype_t>(*dmap[chidx].second);
}

void DMap::slice(std::vector<int> &idcs) {
    for (auto it=begin(); it!=end(); ++it) {
        vec_t *new_v = new vec_t(idcs.size(), 0.); 
        for (int ii=0; ii<idcs.size(); ++ii) {
            (*new_v)[ii] = (*(it->second))[idcs[ii]];
        }
        delete (*it).second;
        (*it).second = new_v;
    }
}

bpy::object DMap::dotOuterNumpy(DMap *other, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    DMapMatrix::matrix_t output(this->size(), other->size(), 0.0);
    this->dotOuter(other, output);
    return npc.ublas_to_numpy<double>(output);
}

void DMap::dotOuter(DMap *other, matrix_t &output) {
    int i = 0;
    int j = 0;
    for (auto it=begin(); it!=end(); ++it, ++i) {
        j = 0;
        for (auto jt=other->begin(); jt!=other->end(); ++jt, ++j) {
            dtype_t r12 = 0.0;
            soap::linalg::linalg_dot(*(it->second), *(jt->second), r12);
            output(i,j) = r12;
        }
    }
}

double DMap::dot(DMap *other) {
    if (other->size() < this->size()) return other->dot(this);
    dtype_t res = 0.0;
    dtype_t r12 = 0.0;
    // NOTE This is clear but inefficient, see improved version below.
    //for (auto it=dmap.begin(); it!=dmap.end(); ++it) {
    //    auto jt = other->dmap.find(it->first);
    //    if (jt != other->end()) {
    //        // Manual version, use linalg_dot for speed
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
    this->add(other, 1.0);
}

void DMap::add(DMap *other, double scale) {
    this->addIgnoreGradients(other, scale);
    this->addGradients(other, scale);
}

void DMap::addIgnoreGradients(DMap *other, double scale) {
    dmap_t add_entries; 
    auto it = this->begin();
    auto jt = other->begin();
    while (jt != other->end()) {
        if (it == this->end() || jt->first < it->first) {
            vec_t *add_vec = new vec_t(jt->second->size());
            *add_vec = *(jt->second)*scale;
            add_entries.push_back(channel_t(jt->first, add_vec));
            ++jt;
        } else if (jt->first == it->first) {
            *(it->second) += *(jt->second)*scale;
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

void DMap::addGradients(DMap *other, double scale) {
    pid_gradmap_t add_entries;
    auto it = this->beginGradients();
    auto jt = other->beginGradients();
    while (jt != other->endGradients()) {
        if (it == this->endGradients() || (*jt)->pid < (*it)->pid) {
            GradMap *add_g = new GradMap((*jt)->pid, "g");
            add_g->add(*jt, scale);
            add_entries.push_back(add_g);
            ++jt;
        } else if ((*jt)->pid == (*it)->pid) {
            (*it)->add(*jt, scale);
            ++it;
            ++jt;
        } else if ((*jt)->pid > (*it)->pid) {
            ++it;
        }
    }
    for (auto it=add_entries.begin(); it!=add_entries.end(); ++it) {
        pid_gradmap.push_back(*it);
    }
    this->sortGradients();
}

void DMap::sort() {
    std::sort(dmap.begin(), dmap.end(), 
        [](DMap::channel_t c1, DMap::channel_t c2) {
            return c1.first <= c2.first;
        }
    );
}

void DMap::sortGradients() {
    std::sort(pid_gradmap.begin(), pid_gradmap.end(),
        [](GradMap *g1, GradMap *g2) {
            return g1->getPid() <= g2->getPid();
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
    for (auto it=pid_gradmap.begin(); it!=pid_gradmap.end(); ++it) {
        (*it)->multiply(c);
    }
}

void DMap::normalize() {
    // Normalization here means the following:
    // Spectrum: X -> X_norm = X/(X.X)**0.5
    // Gradients: G -> G_norm = G' - X_norm * (G'.X_norm)
    // where G' = G/(X.X)**0.5
    double norm = std::sqrt(this->dot(this));
    this->multiply(1./norm); // NOTE This also transforms Gx, Gy, Gz -> Gx', Gy', Gz'
    for (auto it=pid_gradmap.begin(); it!=pid_gradmap.end(); ++it) {
        DMap *gx = (*it)->x();
        DMap *gy = (*it)->y();
        DMap *gz = (*it)->z();
        double sx = this->dot(gx); // = (Gx'.X_norm)
        double sy = this->dot(gy); // = (Gy'.X_norm)
        double sz = this->dot(gz); // = (Gz'.X_norm)
        gx->addIgnoreGradients(this, -sx); // Gx' - X_norm*(Gx'.X_norm)
        gy->addIgnoreGradients(this, -sy); // Gy' - X_norm*(Gy'.X_norm)
        gz->addIgnoreGradients(this, -sz); // Gz' - X_norm*(Gz'.X_norm)
    }
}

void DMap::adapt(AtomicSpectrum *atomic) {
    auto spec = atomic->getXnklMap();
    this->adapt(spec);
    filter = atomic->getCenterType();
    AtomicSpectrum::map_pid_xnkl_t &map_pid_xnkl = atomic->getPowerGradMap();
    if (map_pid_xnkl.size() > 0) {
        bool comoving_center = true; // TODO Expose
        this->adaptPidGradients(
            atomic->getCenterId(), map_pid_xnkl, comoving_center);
    }
    this->normalize();
}

void DMap::adapt(AtomicSpectrum::map_xnkl_t &map_xnkl) {
    for (auto it=map_xnkl.begin(); it!=map_xnkl.end(); ++it) {
        TypeEncoder::code_t e1 = ENCODER.encode(it->first.first);
        TypeEncoder::code_t e2 = ENCODER.encode(it->first.second);
        // Exclude redundant channels (e.g., project C:H, but not H:C)
        if (e1 > e2) continue; 
        auto coeff = it->second->getCoefficients();
        int length = coeff.size1()*coeff.size2();
        auto v = new vec_t(length);
        int c = 0;
        for (int i=0; i<coeff.size1(); ++i) {
            for (int j=0; j<coeff.size2(); ++j, ++c) {
                (*v)(c) = coeff(i,j).real();
            }
        }
        TypeEncoder::code_t e = ENCODER.encode(it->first.first, it->first.second);
        channel_t p(e, v);
        dmap.push_back(p);
    }
    this->sort();
}

void DMap::adaptPidGradients(
        int center_pid, 
        AtomicSpectrum::map_pid_xnkl_t &map_pid_xnkl,
        bool comoving_center) {
    for (auto it=pid_gradmap.begin(); it!=pid_gradmap.end(); ++it) {
        delete *it;
    }
    pid_gradmap.clear();
    bool has_center_grads = false;
    for (auto it=map_pid_xnkl.begin(); it!=map_pid_xnkl.end(); ++it) {
        int pid = it->first;
        if (pid == center_pid) has_center_grads = true;
        AtomicSpectrum::map_xnkl_t &map_xnkl = it->second;
        GradMap *new_gradmap = new GradMap(pid, "g");
        pid_gradmap.push_back(new_gradmap);
        new_gradmap->adapt(map_xnkl);
    }
    if (!has_center_grads && comoving_center) {
        GradMap *cgrad = new GradMap(center_pid, "g");
        for (auto it=beginGradients(); it!=endGradients(); ++it) {
            cgrad->add(*it, -1.);
        }
        pid_gradmap.push_back(cgrad);
    }
    this->sortGradients();
}

void DMap::convolve(int N, int L) {
    dmap_t out;
    for (auto it=begin(); it!=end(); ++it) {
        for (auto jt=it; jt!=end(); ++jt) {
            vec_t &vi = *(it->second);
            vec_t &vj = *(jt->second);
            vec_t *vv = new vec_t(N*N*(L+1), 0.0);
            assert(vi.size() == N*(L+1)*(L+1) && "Vector dimension inconsistent with input.");
            for (int n=0; n<N; ++n) {
                for (int k=0; k<N; ++k) {
                    for (int l=0; l<L+1; ++l) {
                        int nkl = n*N*(L+1) + k*(L+1) + l;
                        dtype_t vvnkl = 0.0;
                        for (int m=-l; m<=l; ++m) {
                            int nlm = n*(L+1)*(L+1) + l*l+l+m;
                            int klm = k*(L+1)*(L+1) + l*l+l+m;
                            vvnkl += vi(nlm)*vj(klm); 
                        }
                        (*vv)(nkl) = vvnkl;
                    }
                }
            }
            TypeEncoder::code_t e = ENCODER.encode(it->first, jt->first);
            channel_t p(e,vv);
            out.push_back(p);
        }
    }
    for (auto it=dmap.begin(); it!=dmap.end(); ++it) {
        delete (*it).second;
    }
    dmap.clear();
    dmap = out;
    this->sort();
    double norm = std::sqrt(this->dot(this));
    this->multiply(1./norm);
}

void DMap::adaptCoherent(AtomicSpectrum *atomic) {
    auto spec = atomic->getQnlmMap();
    for (auto it=spec.begin(); it!=spec.end(); ++it) {
        auto coeff = it->second->getCoefficients();
        int N = coeff.size1();
        int L = int(std::sqrt(coeff.size2())-0.5);
        int length = N*(L+1)*(L+1);
        auto v = new vec_t(length);
        double reco = 1./std::sqrt(2.);
        std::complex<double> imco = std::complex<double>(0.,1.)/std::sqrt(2.);
        for (int n=0; n<N; ++n) {
            int n0 = n*(L+1)*(L+1);
            for (int l=0; l<L+1; ++l) {
                int lm0 = l*l+l;
                dtype_t rnl0 = std::real(coeff(n, lm0));
                (*v)(n0+lm0) = rnl0;
                for (int m=1; m<=l; ++m) {
                    auto qnlm  = coeff(n, lm0+m);
                    auto qnl_m = coeff(n, lm0-m);
                    dtype_t rnl_m = std::real(imco*(qnl_m - std::pow(-1, m)*qnlm));
                    dtype_t rnlm  = std::real(reco*(qnl_m + std::pow(-1, m)*qnlm));
                    (*v)(n0+lm0-m) = rnl_m;
                    (*v)(n0+lm0+m) = rnlm;
                }
            }
        }
        TypeEncoder::code_t e = ENCODER.encode(it->first);
        channel_t p(e, v);
        dmap.push_back(p);
    }
    this->sort();
    filter = atomic->getCenterType();
}

DMap *DMap::dotGradLeft(DMap *other, double coeff, double power, DMap *res) {
    DMap tmp = DMap();
    // Scalar dot product
    vec_t *kv = new vec_t(1);
    dtype_t k = this->dot(other);
    (*kv)(0) = coeff*std::pow(this->dot(other), power);
    TypeEncoder::code_t e = 0;
    tmp.dmap.push_back(channel_t(e, kv));
    // Gradients
    for (auto it=beginGradients(); it!=endGradients(); ++it) {
        DMap *gx = (*it)->x();
        DMap *gy = (*it)->y();
        DMap *gz = (*it)->z();
        dtype_t kx = coeff*power*std::pow(k, power-1)*gx->dot(other);
        dtype_t ky = coeff*power*std::pow(k, power-1)*gy->dot(other);
        dtype_t kz = coeff*power*std::pow(k, power-1)*gz->dot(other);
        GradMap *g = new GradMap((*it)->pid, "g", kx, ky, kz);
        tmp.pid_gradmap.push_back(g);
    }
    // Add to outgoing
    res->add(&tmp);
    return res;
}

void DMap::registerPython() {
    using namespace boost::python;
    dtype_t (DMap::*val_scalar)()
        = &DMap::val;
    bpy::object (DMap::*val_vector)(int, std::string)
        = &DMap::val;
    void (DMap::*onlyAdd)(DMap*)
        = &DMap::add;
    void (DMap::*scaleAdd)(DMap*,double)
        = &DMap::add;

    class_<DMap, DMap*>("DMap", init<>())
        .add_property("filter", &DMap::getFilter)
	    .add_property("gradients", range<return_value_policy<reference_existing_object> >(
            &DMap::beginGradients, &DMap::endGradients))
        .def("listChannels", &DMap::listChannels)
        .def("normalize", &DMap::normalize)
        .def("dot", &DMap::dot)
        .def("dotGradLeft", &DMap::dotGradLeft, return_value_policy<reference_existing_object>())
        .def("dotOuter", &DMap::dotOuterNumpy)
        .def("add", onlyAdd)
        .def("add", scaleAdd)
        .def("val", val_scalar)
        .def("val", val_vector)
        .def("multiply", &DMap::multiply)
        .def("convolve", &DMap::convolve)
        .def("dotFilter", &DMap::dotFilter);

    class_<DMap::pid_gradmap_t>("GradMapVector")
        .def(vector_indexing_suite<DMap::pid_gradmap_t>());
}

GradMap::GradMap() 
        : pid(-1), filter("") {
    ;
}

GradMap::GradMap(int particle_id, std::string filter_type) 
        : pid(particle_id), filter(filter_type) {
    this->zero();
}

GradMap::~GradMap() {
    this->clear();
}

void GradMap::clear() {
    for (auto it=gradmap.begin(); it!=gradmap.end(); ++it) {
        delete (*it);
    }
    gradmap.clear();
}

void GradMap::zero() {
    this->clear();
    DMap *dx = new DMap(filter);
    DMap *dy = new DMap(filter);
    DMap *dz = new DMap(filter);
    gradmap.push_back(dx);
    gradmap.push_back(dy);
    gradmap.push_back(dz);
}

void GradMap::multiply(double c) {
    for (auto it=gradmap.begin(); it!=gradmap.end(); ++it) {
        (*it)->multiply(c);
    }
}

void GradMap::add(GradMap *other, double c) {
    //assert(other->pid == pid);
    assert(other->gradmap.size() == gradmap.size());
    for (int i=0; i<gradmap.size(); ++i) {
        this->get(i)->addIgnoreGradients(other->get(i), c);
    }
}

GradMap::GradMap(int particle_id, std::string filter_type, double gx, double gy, double gz)
        : pid(particle_id), filter(filter_type) {
    this->zero();
    DMap *dx = this->get(0);
    DMap *dy = this->get(1);
    DMap *dz = this->get(2);
    DMap::vec_t *vgx = new DMap::vec_t(1);
    DMap::vec_t *vgy = new DMap::vec_t(1);
    DMap::vec_t *vgz = new DMap::vec_t(1);
    (*vgx)(0) = gx;
    (*vgy)(0) = gy;
    (*vgz)(0) = gz;
    DMap::channel_t px(0, vgx);
    DMap::channel_t py(0, vgy);
    DMap::channel_t pz(0, vgz);
    dx->dmap.push_back(px);
    dy->dmap.push_back(py);
    dz->dmap.push_back(pz);
}

void GradMap::adapt(AtomicSpectrum::map_xnkl_t &map_xnkl) {
    this->zero();
    DMap *dx = this->get(0);
    DMap *dy = this->get(1);
    DMap *dz = this->get(2);
    for (auto it=map_xnkl.begin(); it!=map_xnkl.end(); ++it) {
        TypeEncoder::code_t e1 = ENCODER.encode(it->first.first);
        TypeEncoder::code_t e2 = ENCODER.encode(it->first.second);
        if (e1 > e2) continue; 
        auto coeff_x = it->second->getCoefficientsGradX();
        auto coeff_y = it->second->getCoefficientsGradY();
        auto coeff_z = it->second->getCoefficientsGradZ();
        int length = coeff_x.size1()*coeff_x.size2();
        auto vx = new DMap::vec_t(length);
        auto vy = new DMap::vec_t(length);
        auto vz = new DMap::vec_t(length);
        int c = 0;
        for (int i=0; i<coeff_x.size1(); ++i) {
            for (int j=0; j<coeff_x.size2(); ++j, ++c) {
                (*vx)(c) = coeff_x(i,j).real();
                (*vy)(c) = coeff_y(i,j).real();
                (*vz)(c) = coeff_z(i,j).real();
            }
        }
        TypeEncoder::code_t e = ENCODER.encode(it->first.first, it->first.second);
        DMap::channel_t px(e, vx);
        DMap::channel_t py(e, vy);
        DMap::channel_t pz(e, vz);
        dx->dmap.push_back(px);
        dy->dmap.push_back(py);
        dz->dmap.push_back(pz);
    }
    dx->sort();
    dy->sort();
    dz->sort();
}

void GradMap::registerPython() {
    using namespace boost::python;
    class_<GradMap, GradMap*>("GradMap", init<int, std::string>())
        .add_property("pid", &GradMap::getPid)
        .add_property("x", &GradMap::xval)
        .add_property("y", &GradMap::yval)
        .add_property("z", &GradMap::zval)
        .def("__getitem__", &GradMap::get, return_value_policy<reference_existing_object>())
        .def("getX", &GradMap::x, return_value_policy<reference_existing_object>())
        .def("getY", &GradMap::y, return_value_policy<reference_existing_object>())
        .def("getZ", &GradMap::z, return_value_policy<reference_existing_object>());
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

void DMapMatrix::append(DMap *dmap_arg) {
    DMap *new_dmap = new DMap(dmap_arg->filter);
    new_dmap->dmap = dmap_arg->dmap;
    dmm.push_back(new_dmap);
}

void DMapMatrix::append(Spectrum *spectrum) {
    for (auto it=spectrum->beginAtomic(); it!=spectrum->endAtomic(); ++it) {
        DMap *new_dmap = new DMap();
        new_dmap->adapt(*it);
        dmm.push_back(new_dmap);
    }
}

void DMapMatrix::appendCoherent(Spectrum *spectrum) {
    for (auto it=spectrum->beginAtomic(); it!=spectrum->endAtomic(); ++it) {
        DMap *new_dmap = new DMap();
        new_dmap->adaptCoherent(*it);
        dmm.push_back(new_dmap);
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

void DMapMatrix::normalize() {
    for (auto it=begin(); it!=end(); ++it) {
        (*it)->normalize();
    }
}

void DMapMatrix::convolve(int N, int L) {
    for (auto it=begin(); it!=end(); ++it) {
        (*it)->convolve(N, L);
    }
}

void DMapMatrix::slicePython(bpy::list &py_idcs) {
    std::vector<int> idcs;
    for (int i=0; i<bpy::len(py_idcs); ++i) {
        idcs.push_back(bpy::extract<int>(py_idcs[i]));
    }
    this->slice(idcs);
}

void DMapMatrix::slice(std::vector<int> &idcs) {
    for (auto it=begin(); it!=end(); ++it) {
        (*it)->slice(idcs);
    }
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

std::string DMapMatrix::dumps() {
    std::stringstream ss;
    boost::archive::binary_oarchive arch(ss);
    arch << (*this);
    return ss.str();
}

void DMapMatrix::loads(std::string pstr) {
    std::stringstream ss;
    ss << pstr;
    boost::archive::binary_iarchive arch(ss);
    arch >> (*this);
    return;
}

void DMapMatrix::registerPython() {
    using namespace boost::python;
    void (DMapMatrix::*appendDMap)(DMap*)
        = &DMapMatrix::append;
    void (DMapMatrix::*appendSpectrum)(Spectrum*)
        = &DMapMatrix::append;
    class_<DMapMatrix, DMapMatrix*>("DMapMatrix", init<>())
        .def(init<std::string>())
        .add_property("rows", &DMapMatrix::rows)
        .def("__len__", &DMapMatrix::rows)
        .def("__getitem__", &DMapMatrix::getRow, return_value_policy<reference_existing_object>())
        .def("addView", &DMapMatrix::addView)
        .def("getView", &DMapMatrix::getView, return_value_policy<reference_existing_object>())
        .def("append", appendDMap)
        .def("append", appendSpectrum)
        .def("appendCoherent", &DMapMatrix::appendCoherent)
        .def("sum", &DMapMatrix::sum)
        .def("slice", &DMapMatrix::slicePython)
        .def("normalize", &DMapMatrix::normalize)
        .def("convolve", &DMapMatrix::convolve)
        .def("dot", &DMapMatrix::dotNumpy)
        .def("dotFilter", &DMapMatrix::dotFilterNumpy)
        .def("dumps", &DMapMatrix::dumps)
        .def("loads", &DMapMatrix::loads)
        .def("load", &DMapMatrix::load)
        .def("save", &DMapMatrix::save);
}

DMapMatrixSet::DMapMatrixSet() : is_view(false) {
    ;
}

DMapMatrixSet::DMapMatrixSet(bool set_as_view) : is_view(set_as_view) {
    ;
}

DMapMatrixSet::DMapMatrixSet(std::string archfile) : is_view(false) {
    this->load(archfile);
}

DMapMatrixSet::~DMapMatrixSet() {
    if (!is_view) {
        for (auto it=begin(); it!=end(); ++it) delete *it;
    }
    dset.clear();
    for (auto it=views.begin(); it!=views.end(); ++it) delete *it;
    views.clear();
}

void DMapMatrixSet::clear() {
    if (!is_view) {
        for (auto it=begin(); it!=end(); ++it) delete *it;
    }
    dset.clear();
    for (auto it=views.begin(); it!=views.end(); ++it) delete *it;
    views.clear();
}

DMapMatrixSet *DMapMatrixSet::getView(boost::python::list idcs) {
    DMapMatrixSet *view = new DMapMatrixSet(false);
    for (int i=0; i<boost::python::len(idcs); ++i) {
        int idx = boost::python::extract<int>(idcs[i]);
        if (idx-1 > size()) throw soap::base::OutOfRange(
            "Index " + lexical_cast<std::string>(idx, ""));
        view->append(dset[idx]); 
    }
    return view;
}

void DMapMatrixSet::append(DMapMatrix *dmap) {
    dset.push_back(dmap);
}

void DMapMatrixSet::extend(DMapMatrixSet *other) {
    if (this->is_view) throw soap::base::SanityCheckFailed(
        "Extending matrix view not permitted.");
    for (auto it=other->begin(); it!=other->end(); ++it) {
        dset.push_back(*it);
    }
    other->is_view = true;
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

void DMapMatrixSet::slicePython(bpy::list &py_idcs) {
    std::vector<int> idcs;
    for (int i=0; i<bpy::len(py_idcs); ++i) {
        idcs.push_back(bpy::extract<int>(py_idcs[i]));
    }
    this->slice(idcs);
}

void DMapMatrixSet::slice(std::vector<int> &idcs) {
    for (auto it=begin(); it!=end(); ++it) {
        (*it)->slice(idcs);
    }
}

void DMapMatrixSet::registerPython() {
    using namespace boost::python;
    class_<DMapMatrixSet, DMapMatrixSet*>("DMapMatrixSet", init<>())
        .def(init<std::string>())
        .def("__len__", &DMapMatrixSet::size)
        .def("__getitem__", &DMapMatrixSet::get, return_value_policy<reference_existing_object>())
        .def("__getitem__", &DMapMatrixSet::getView, return_value_policy<reference_existing_object>())
        .add_property("size", &DMapMatrixSet::size)
        .def("save", &DMapMatrixSet::save)
        .def("load", &DMapMatrixSet::load)
        .def("slice", &DMapMatrixSet::slicePython)
        .def("clear", &DMapMatrixSet::clear)
        .def("append", &DMapMatrixSet::append)
        .def("extend", &DMapMatrixSet::extend);
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

boost::python::object BlockLaplacian::getBlockNumpy(int idx, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    return npc.ublas_to_numpy<dtype_t>(*blocks[idx]);
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
        .def("__getitem__", &BlockLaplacian::getItemNumpy)
        .def("dot", &BlockLaplacian::dotNumpy)
        .def("append", &BlockLaplacian::appendNumpy)
        .def("getBlock", &BlockLaplacian::getBlockNumpy)
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

TypeEncoder::code_t TypeEncoder::encode(code_t code1, code_t code2) {
    return code1*size() + code2;
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
