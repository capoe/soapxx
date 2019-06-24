#ifndef _SOAP_DMAP_KERNEL_HPP
#define _SOAP_DMAP_KERNEL_HPP

#include "soap/base/objectfactory.hpp"
#include "soap/dmap.hpp"

namespace soap {

class BaseKernel
{
  public:
    BaseKernel() {;}
    virtual std::string identify() { return "basekernel"; }
    virtual void configure(Options &options) {;}
    virtual ~BaseKernel() {;}
    virtual double evaluate(DMapMatrix*, DMapMatrix*, DMapMatrix::matrix_t &K_out) = 0;
};

class TopKernel
{
  public:
    TopKernel() {;}
    virtual std::string identify() { return "topkernel"; }
    virtual ~TopKernel() {;}
    virtual void configure(Options &options) {;}
    double evaluateNumpy(boost::python::object &np_K, std::string np_dtype);
    virtual double evaluate(DMapMatrix::matrix_t &K) = 0;
    virtual void attributeLeft(DMapMatrix::matrix_t &K, 
        DMapMatrix::matrix_t &K_out, int i_off, int j_off) = 0;
};

class Kernel
{
  public:
    typedef Options metadata_t;
    typedef ub::matrix<double> output_t;
    Kernel(Options &options);
    ~Kernel();
    metadata_t *getMetadata() { return metadata; }
    boost::python::object getOutput(int slot, std::string np_type);
    void clearOutput();
    void clearThenAllocateOutput(int n_rows, int n_cols);
    int outputSlots() { return kernelmats_out.size(); }
    boost::python::object evaluatePython(
        DMapMatrixSet *dset1,
        DMapMatrixSet *dset2,
        bool symmetric,
        std::string np_type);
    void evaluate(
        DMapMatrixSet *dset1,
        DMapMatrixSet *dset2,
        bool symmetric,
        DMapMatrix::matrix_t &output);
    double evaluate(
        DMapMatrix *dmap1,
        DMapMatrix *dmap2);
    void evaluateAll(
        DMapMatrixSet *dset1,
        DMapMatrixSet *dset2,
        bool symmetric);
    double evaluateTopkernel(
        boost::python::object &np_K, 
        std::string np_dtype);
    boost::python::object attributeLeftPython(
        DMapMatrix *dmap1,
        DMapMatrixSet *dset2,
        std::string np_type);
    void attributeLeft(
        DMapMatrix *dmap1,
        DMapMatrixSet *dset2,
        DMapMatrix::matrix_t &output);
    void addTopkernel(Options &options);
    static void registerPython();
  private:
    BaseKernel *basekernel;
    metadata_t *metadata;
    std::vector<TopKernel*> topkernels;
    std::vector<output_t*> kernelmats_out;
};

class BaseKernelDot : public BaseKernel
{
  public:
    BaseKernelDot();
    void configure(Options &options);
    double evaluate(DMapMatrix*, DMapMatrix*, DMapMatrix::matrix_t &K_out);
  private:
    double exponent;
    bool filter;
};

class TopKernelRematch : public TopKernel
{
  public:
    TopKernelRematch();
    void configure(Options &options);
    double evaluate(DMapMatrix::matrix_t &K);
    void attributeLeft(DMapMatrix::matrix_t &K, 
        DMapMatrix::matrix_t &K_out, int i_off, int j_off);
  private:
    double gamma; // Rematch temperature
    double eps;   // Convergence tolerance
    double omega; // Mixing factor, successive overrelaxation
};

class TopKernelCanonical : public TopKernel
{
  public:
    TopKernelCanonical();
    void configure(Options &options);
    double evaluate(DMapMatrix::matrix_t &K);
    void attributeLeft(DMapMatrix::matrix_t &K, 
        DMapMatrix::matrix_t &K_out, int i_off, int j_off);
  private:
    double beta;
};

class TopKernelAverage : public TopKernel
{
  public:
    TopKernelAverage();
    void configure(Options &options);
    double evaluate(DMapMatrix::matrix_t &K);
    void attributeLeft(DMapMatrix::matrix_t &K, 
        DMapMatrix::matrix_t &K_out, int i_off, int j_off);
  private:
};

class KernelInterface
{
  public:
    KernelInterface() {;}
    static void registerPython();
  private:
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
        throw std::runtime_error("Factory key '" + key + "' not found.");
    }
}

}

#endif
