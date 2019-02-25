#include "soap/kernel.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

double TopKernel::evaluateNumpy(boost::python::object &np_K, std::string np_dtype) {
    DMapMatrix::matrix_t K;
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    npc.numpy_to_ublas<DMapMatrix::dtype_t>(np_K, K);
    return this->evaluate(K);
}

TopKernelRematch::TopKernelRematch() : gamma(0.01), eps(1e-6), omega(1.2) {;}

void TopKernelRematch::configure(Options &options) {
    GLOG() << "Configuring top kernel (rematch)" << std::endl;
    gamma = options.get<double>("rematch_gamma");
    eps = options.get<double>("rematch_eps");
    omega = options.get<double>("rematch_omega");
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

TopKernelCanonical::TopKernelCanonical() {
    ;
}

void TopKernelCanonical::configure(Options &options) {
    beta = options.get<double>("canonical_beta");
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

TopKernelAverage::TopKernelAverage() {;}

void TopKernelAverage::configure(Options &options) {
    ;
}

double TopKernelAverage::evaluate(DMapMatrix::matrix_t &K) {
    double k = 0.0;
    for (int i=0; i<K.size1(); ++i)
        for (int j=0; j<K.size2(); ++j)
            k += K(i,j);
    return k;
}

void BaseKernelDot::configure(Options &options) {
    assert(false); // TODO Not yet used, dmm_inner_product is default
}

double BaseKernelDot::evaluate(DMap *a1, DMap *a2) {
    assert(false);
    return 0.0;
}

Kernel::~Kernel() {
    delete basekernel;
    delete topkernel;
    delete metadata;
}

Kernel::Kernel(Options &options) {
    if (options.hasKey("basekernel_type")) { // TODO Not yet used, dmm_inner_product is default
        basekernel = BaseKernelCreator().create(options.get<std::string>("basekernel_type"));
        basekernel->configure(options);
    }
    topkernel = TopKernelCreator().create(options.get<std::string>("topkernel_type"));
    topkernel->configure(options);
}

boost::python::object Kernel::evaluatePython(DMapMatrixSet *dset1, DMapMatrixSet *dset2, 
        double power, bool filter, bool symmetric, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    DMapMatrix::matrix_t output(dset1->size(), dset2->size(), 0.0);
    this->evaluate(dset1, dset2, power, filter, symmetric, output);
    return npc.ublas_to_numpy<double>(output);
}

void Kernel::evaluate(DMapMatrixSet *dset1, DMapMatrixSet *dset2, 
        double power, bool filter, bool symmetric, DMapMatrix::matrix_t &output) {
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
            // TODO Add base kernel here
            dmm_inner_product(*(*it), *(*jt), power, filter, Kij);
            output(i_row, j_col) = topkernel->evaluate(Kij);
        }
    }
    GLOG() << std::endl;
}

double Kernel::evaluateTop(boost::python::object &np_K, std::string np_dtype) {
    return topkernel->evaluateNumpy(np_K, np_dtype);
}

void Kernel::registerPython() {
    using namespace boost::python;
    class_<Kernel, Kernel*>("Kernel", init<Options&>())
        .def("evaluateTop", &Kernel::evaluateTop)
        .def("getMetadata", &Kernel::getMetadata, return_value_policy<reference_existing_object>())
        .def("evaluate", &Kernel::evaluatePython);
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
