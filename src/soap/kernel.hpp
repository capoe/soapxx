#ifndef _SOAP_KERNEL_HPP
#define _SOAP_KERNEL_HPP

#include <soap/spectrum.hpp>

#include "soap/base/objectfactory.hpp"
namespace soap {

class BaseKernel
{
public:
    BaseKernel() {;}
    virtual std::string identify() { return "basekernel"; }
    virtual void configure(Options &options) {;}
    virtual ~BaseKernel() {;}
    virtual double evaluate(AtomicSpectrum*, AtomicSpectrum*) = 0;
};

class TopKernel
{
public:
    TopKernel() {;}
    virtual std::string identify() { return "topkernel"; }
    virtual ~TopKernel() {;}
    virtual void configure(Options &options) {;}
    virtual double evaluate(Spectrum*, Spectrum*, BaseKernel*) = 0;
};

class Kernel
{
public:
    Kernel(Options &options);
    ~Kernel();
    double evaluate(Spectrum*, Spectrum*);
    static void registerPython();
private:
    BaseKernel *basekernel;
    TopKernel *topkernel;
};

class BaseKernelDot : public BaseKernel
{
public:
    BaseKernelDot() {;}
    void configure(Options &options);
    double evaluate(AtomicSpectrum*, AtomicSpectrum*);
private:
    double exponent;
    double coefficient;
};

class TopKernelRematch : public TopKernel
{
public:
    TopKernelRematch() {;}
    void configure(Options &options);
    double evaluate(Spectrum*, Spectrum*, BaseKernel*);
private:
    double gamma;
};

//template
//<
//    typename KernelFunction // delta^2 * (IX * X)^xi ; gradient
//    typename Adaptor // Spectrum (select centers, mu, nu) -> {X}
//>
//class KernelPotential : KernelPotentialBase
//{
//public:
//
//    void acquire(Spectrum *spectrum, double weight) {
//        // For X in spectrum: adapt X, store (extend _IX, _alpha)
//    }
//    void evaluateForce(Structure *) {
//        // Structure -> Spectrum -> adapt X -> Kernel function -> force
//    }
//    void evaluateEnergy(Structure *) {
//        // Structure -> Spectrum -> adapt X -> Kernel function -> energy
//    }
//
//private:
//
//};
//
//class KernelPotentialBase
//{
//public:
//    KernelPotential(Basis *, Options *);
//
//    void acquire(Spectrum *spectrum, double weight);
//    void evaluateForce(Structure *); // Structure -> Spectrum -> X -> Kernel force
//
//private:
//
//    ub::vector<double> _alpha;
//    ub::matrix<double> _IX;
//    double _xi;
//    double _delta;
//
//    Basis *_basis;
//    Options *_options;
//};

class BaseKernelFactory
    : public soap::base::ObjectFactory<std::string, BaseKernel>
{
private:
    BaseKernelFactory() {}
public:
    static void registerAll(void);
    BaseKernel *create(const std::string &key);
    friend BaseKernelFactory &BaseKernelCreator();
};

inline BaseKernelFactory &BaseKernelCreator() {
    static BaseKernelFactory _instance;
    return _instance;
}

inline BaseKernel *BaseKernelFactory::create(const std::string &key) {
    assoc_map::const_iterator it(getObjects().find(key));
    if (it != getObjects().end()) {
        BaseKernel *kernel = (it->second)();
        return kernel;
    }
    else {
        throw std::runtime_error("Factory key " + key + " not found.");
    }
}

class TopKernelFactory
    : public soap::base::ObjectFactory<std::string, TopKernel>
{
private:
    TopKernelFactory() {}
public:
    static void registerAll(void);
    TopKernel *create(const std::string &key);
    friend TopKernelFactory &TopKernelCreator();
};

inline TopKernelFactory &TopKernelCreator() {
    static TopKernelFactory _instance;
    return _instance;
}

inline TopKernel *TopKernelFactory::create(const std::string &key) {
    assoc_map::const_iterator it(getObjects().find(key));
    if (it != getObjects().end()) {
        TopKernel *kernel = (it->second)();
        return kernel;
    }
    else {
        throw std::runtime_error("Factory key " + key + " not found.");
    }
}


}

#endif
