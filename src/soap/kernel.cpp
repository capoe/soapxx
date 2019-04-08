#include "soap/kernel.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"
#include "soap/base/tokenizer.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

double TopKernel::evaluateNumpy(boost::python::object &np_K, std::string np_dtype) {
    DMapMatrix::matrix_t K;
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    npc.numpy_to_ublas<DMapMatrix::dtype_t>(np_K, K);
    return this->evaluate(K);
}

TopKernelRematch::TopKernelRematch() : gamma(0.05), eps(1e-6), omega(1.0) {;}

void TopKernelRematch::configure(Options &options) {
    gamma = options.get<double>("rematch_gamma");
    eps = options.get<double>("rematch_eps");
    omega = options.get<double>("rematch_omega");
    GLOG() << "Configuring top kernel (rematch)";
    GLOG() << " gamma=" << gamma << " eps=" << eps << " omega=" << omega << std::endl;
}

double TopKernelRematch::evaluate(DMapMatrix::matrix_t &K) {
    int nx = K.size1();
    int ny = K.size2();
    DMapMatrix::vec_t u(nx, 1.0);
    DMapMatrix::vec_t u_in(nx, 1.0);
    DMapMatrix::vec_t du(nx);
    DMapMatrix::vec_t v(ny, 1.0);
    DMapMatrix::vec_t v_in(ny, 1.0);
    double ax = 1./nx;
    double ay = 1./ny;
    double lambda = 1./gamma;
    DMapMatrix::matrix_t Kg(nx,ny);
    for (int i=0; i<nx; ++i)
        for (int j=0; j<ny; ++j)
            Kg(i,j) = std::exp(-(1-K(i,j))*lambda);
    int i_iter = 0;
    double err = 0.0;
    // TODO Add convergence flag and send to user
    while (true) {
        soap::linalg::linalg_matrix_vector_dot(Kg, v, u, false, 1.0, 0.0);
        for (int i=0; i<nx; ++i) u(i)= omega*ax/u(i) + (1-omega)*u_in(i);
        soap::linalg::linalg_matrix_vector_dot(Kg, u, v, true, 1.0, 0.0);
        for (int i=0; i<ny; ++i) v(i)= omega*ay/v(i) + (1-omega)*v_in(i);
        // Check convergence
        du = u-u_in;
        soap::linalg::linalg_dot(du, du, err);
        err = std::sqrt(err/nx);
        if (err < eps) break;
        // Step
        u_in = u;
        v_in = v;
        i_iter += 1;
    }
    double k = 0.0;
    ub::matrix<double> Pij(nx, ny, 0.0);
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            Pij(i,j) = u(i)*Kg(i,j)*v(j);
            k += Pij(i,j)*K(i,j);
        }
    }
    return k;
}

void TopKernelRematch::attributeLeft(DMapMatrix::matrix_t &K, 
        DMapMatrix::matrix_t &K_out, int i_off, int j_off) {

}

TopKernelCanonical::TopKernelCanonical() : beta(0.5) {
    ;    
}

void TopKernelCanonical::configure(Options &options) {
    beta = options.get<double>("canonical_beta");
    GLOG() << "Configuring top kernel (canonical)";
    GLOG() << " beta=" << beta << std::endl;
}

double TopKernelCanonical::evaluate(DMapMatrix::matrix_t &K) {
    int nx = K.size1();
    int ny = K.size2();
    double lambda = 1./beta;
    DMapMatrix::matrix_t P(nx,ny);
    DMapMatrix::vec_t zx(nx, 0.0);
    DMapMatrix::vec_t zy(ny, 0.0);
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            P(i,j) = std::exp(lambda*K(i,j));
            zx(i) += P(i,j);
            zy(j) += P(i,j);
        }
    }
    double k = 0.0;
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            k += K(i,j)*std::pow(P(i,j), 2)/(zx(i)*zy(j));
        }
    }
    return k;
}

void TopKernelCanonical::attributeLeft(DMapMatrix::matrix_t &K, 
        DMapMatrix::matrix_t &K_out, int i_off, int j_off) {

}

TopKernelAverage::TopKernelAverage() {;}

void TopKernelAverage::configure(Options &options) {
    GLOG() << "Configuring top kernel (average):" << std::endl;
}

double TopKernelAverage::evaluate(DMapMatrix::matrix_t &K) {
    double k = 0.0;
    for (int i=0; i<K.size1(); ++i)
        for (int j=0; j<K.size2(); ++j)
            k += K(i,j);
    return k;
}

void TopKernelAverage::attributeLeft(DMapMatrix::matrix_t &K, 
        DMapMatrix::matrix_t &K_out, int i_off, int j_off) {
    for (int i=0; i<K.size1(); ++i) {
        double ki = 0.0;
        for (int j=0; j<K.size2(); ++j) {
            ki += K(i,j);
        }
        K_out(i_off+i, j_off) = ki;
    }
}

BaseKernelDot::BaseKernelDot() : exponent(2.), filter(false) {
    ;
}

void BaseKernelDot::configure(Options &options) {
    exponent = options.get<double>("base_exponent");
    filter = options.get<bool>("base_filter");
    GLOG() << "Configuring base kernel (dot):";
    GLOG() << " exponent=" << exponent << " filter=" << filter << std::endl;
}

double BaseKernelDot::evaluate(DMapMatrix *m1, DMapMatrix *m2, DMapMatrix::matrix_t &K_out) {
    dmm_inner_product(*m1, *m2, exponent, filter, K_out);
}

Kernel::~Kernel() {
    delete basekernel;
    delete metadata;
    for (auto it=topkernels.begin(); it!=topkernels.end(); ++it) delete *it;
    topkernels.clear();
    this->clearOutput();
}

Kernel::Kernel(Options &options) {
    basekernel = BaseKernelCreator().create(options.get<std::string>("basekernel_type"));
    basekernel->configure(options);
    if (options.get<std::string>("topkernel_type") != "") {
        this->addTopkernel(options);
    }
}

void Kernel::addTopkernel(Options &options) {
    auto topkernel_keys = soap::base::Tokenizer(
        options.get<std::string>("topkernel_type"), ";").ToVector();
    for (auto key : topkernel_keys) {
        TopKernel *topkernel = TopKernelCreator().create(key);
        topkernel->configure(options);
        topkernels.push_back(topkernel);
    }
}

boost::python::object Kernel::evaluatePython(DMapMatrixSet *dset1, DMapMatrixSet *dset2, 
        bool symmetric, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    DMapMatrix::matrix_t output(dset1->size(), dset2->size(), 0.0);
    this->evaluate(dset1, dset2, symmetric, output);
    return npc.ublas_to_numpy<DMapMatrix::dtype_t>(output);
}

void Kernel::evaluate(DMapMatrixSet *dset1, DMapMatrixSet *dset2, 
        bool symmetric, DMapMatrix::matrix_t &output) {
    if (basekernel == NULL || topkernels.size() == 0)
        throw soap::base::SanityCheckFailed("Kernel object not initialized");
    if (symmetric) dset2 = dset1;
    int n_rows = dset1->size();
    int n_cols = dset2->size();
    assert(output.size1() == n_rows && output.size2() == n_cols 
        && "Inconsistent output matrix dimensions");
    int i_row = 0;
    for (auto it=dset1->begin(); it!=dset1->end(); ++it, ++i_row) {
        GLOG() << "\r" << "Row " << i_row+1 << "/" << n_rows << std::flush;
        int j_col = (symmetric) ? i_row : 0;
        for (auto jt=(symmetric) ? it : dset2->begin(); jt!=dset2->end(); ++jt, ++j_col) {
            DMapMatrix::matrix_t Kij((*it)->rows(), (*jt)->rows());
            //dmm_inner_product(*(*it), *(*jt), power, filter, Kij);
            basekernel->evaluate(*it, *jt, Kij);
            output(i_row, j_col) = topkernels[0]->evaluate(Kij);
        }
    }
    GLOG() << std::endl;
}

boost::python::object Kernel::getOutput(int slot, std::string np_dtype) {
    if (slot > kernelmats_out.size()) 
        throw soap::base::OutOfRange("Output slot "+lexical_cast<std::string>(slot, ""));
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    return npc.ublas_to_numpy<DMapMatrix::dtype_t>(*kernelmats_out[slot]);
}

void Kernel::clearOutput() {
    for (auto it=kernelmats_out.begin(); it!=kernelmats_out.end(); ++it) delete *it;
    kernelmats_out.clear();
}

void Kernel::clearThenAllocateOutput(int n_rows, int n_cols) {
    this->clearOutput();
    for (auto it=topkernels.begin(); it!=topkernels.end(); ++it) {
        DMapMatrix::matrix_t *out_i = new DMapMatrix::matrix_t(n_rows, n_cols, 0.0);
        kernelmats_out.push_back(out_i);
    }
}

void Kernel::evaluateAll(DMapMatrixSet *dset1, DMapMatrixSet *dset2, 
        bool symmetric) {
    if (basekernel == NULL || topkernels.size() == 0)
        throw soap::base::SanityCheckFailed("Kernel object not initialized");
    if (symmetric) dset2 = dset1;
    int n_rows = dset1->size();
    int n_cols = dset2->size();
    this->clearThenAllocateOutput(n_rows, n_cols);
    int i_row = 0;
    for (auto it=dset1->begin(); it!=dset1->end(); ++it, ++i_row) {
        GLOG() << "\r" << "Row " << i_row+1 << "/" << n_rows << std::flush;
        int j_col = (symmetric) ? i_row : 0;
        for (auto jt=(symmetric) ? it : dset2->begin(); jt!=dset2->end(); ++jt, ++j_col) {
            DMapMatrix::matrix_t Kij((*it)->rows(), (*jt)->rows());
            //dmm_inner_product(*(*it), *(*jt), power, filter, Kij);
            basekernel->evaluate(*it, *jt, Kij);
            for (int k=0; k<topkernels.size(); ++k) {
                (*kernelmats_out[k])(i_row, j_col) = topkernels[k]->evaluate(Kij);
            }
        }
    }
    GLOG() << std::endl;
}

double Kernel::evaluateTopkernel(boost::python::object &np_K, std::string np_dtype) {
    return topkernels[0]->evaluateNumpy(np_K, np_dtype);
}

boost::python::object Kernel::attributeLeftPython(DMapMatrix *dmap1, DMapMatrixSet *dset2, 
        std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    DMapMatrix::matrix_t output(dmap1->rows(), dset2->size(), 0.0);
    this->attributeLeft(dmap1, dset2, output);
    return npc.ublas_to_numpy<DMapMatrix::dtype_t>(output);
}

void Kernel::attributeLeft(DMapMatrix *dmap1, DMapMatrixSet *dset2, 
        DMapMatrix::matrix_t &output) {
    int n_rows = dmap1->rows();
    int n_cols = dset2->size();
    assert(output.size1() == n_rows && output.size2() == n_cols 
        && "Inconsistent output matrix dimensions");
    int j_col = 0;
    for (auto jt=dset2->begin(); jt!=dset2->end(); ++jt, ++j_col) {
        GLOG() << "\r" << "Col " << j_col+1 << "/" << n_cols << std::flush;
        DMapMatrix::matrix_t Kij(dmap1->rows(), (*jt)->rows());
        basekernel->evaluate(dmap1, *jt, Kij);
        int i_off = 0;
        int j_off = j_col;
        topkernels[0]->attributeLeft(Kij, output, i_off, j_off);
    }
    GLOG() << std::endl;
}

void Kernel::registerPython() {
    using namespace boost::python;
    class_<Kernel, Kernel*>("Kernel", init<Options&>())
        .def("addTopkernel", &Kernel::addTopkernel)
        .def("evaluateTop", &Kernel::evaluateTopkernel)
        .def("getMetadata", &Kernel::getMetadata, return_value_policy<reference_existing_object>())
        .add_property("n_output", &Kernel::outputSlots)
        .def("getOutput", &Kernel::getOutput)
        .def("attributeLeft", &Kernel::attributeLeftPython)
        .def("evaluate", &Kernel::evaluatePython)
        .def("evaluateAll", &Kernel::evaluateAll);
}

void BaseKernelFactory::registerAll(void) {
	BaseKernelCreator().Register<BaseKernelDot>("dot");
}

void TopKernelFactory::registerAll(void) {
    TopKernelCreator().Register<TopKernelAverage>("average");
    TopKernelCreator().Register<TopKernelCanonical>("canonical");
    TopKernelCreator().Register<TopKernelRematch>("rematch");
}

void KernelInterface::registerPython() {
    using namespace boost::python;
    //def("evaluate_kernel", evaluate_kernel_numpy); // NOTE Deprecated
}

}
