#include "soap/kernel.hpp"

namespace soap {

namespace ub = boost::numeric::ublas;

double TopKernelRematch::evaluate(Spectrum *s1, Spectrum *s2, BaseKernel *basekernel) {
    ub::matrix<double> kmat(s1->length(), s2->length());
    int i = 0;
    for (auto it1=s1->beginAtomic(); it1!=s1->endAtomic(); ++it1, ++i) {
        int j = 0;
        for (auto it2=s2->beginAtomic(); it2!=s2->endAtomic(); ++it2, ++j) {
            kmat(i,j) = basekernel->evaluate(*it1, *it2);
            if (i == j) kmat(i,j) = 1.0;
        }
    }
    return 0.0;
}

void TopKernelRematch::configure(Options &options) {
    GLOG() << "Configuring top kernel (rematch)" << std::endl;
    gamma = options.get<double>("topkernel.rematch.gamma");
}

double BaseKernelDot::evaluate(AtomicSpectrum *a1, AtomicSpectrum *a2) {
    return 0.0;
}

void BaseKernelDot::configure(Options &options) {
    GLOG() << "Configuring base kernel (dot)" << std::endl;
    exponent = options.get<double>("basekernel.dot.exponent");
    coefficient = options.get<double>("basekernel.dot.coefficient");

}

Kernel::Kernel(Options &options) {
    basekernel = BaseKernelCreator().create(options.get<std::string>("basekernel.type"));
    basekernel->configure(options);
    topkernel = TopKernelCreator().create(options.get<std::string>("topkernel.type"));
    topkernel->configure(options);
}

Kernel::~Kernel() {
    delete basekernel;
    delete topkernel;
}

double Kernel::evaluate(Spectrum *s1, Spectrum *s2) {
    return topkernel->evaluate(s1, s2, basekernel);
}

void Kernel::registerPython() {
        using namespace boost::python;
        class_<Kernel, Kernel*>("Kernel", init<Options&>())
            .def("evaluate", &Kernel::evaluate);
}

void BaseKernelFactory::registerAll(void) {
	BaseKernelCreator().Register<BaseKernelDot>("dot");
}

void TopKernelFactory::registerAll(void) {
    TopKernelCreator().Register<TopKernelRematch>("rematch");
}

}
