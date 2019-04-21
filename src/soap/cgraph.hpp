#ifndef SOAP_CGRAPH_HPP
#define SOAP_CGRAPH_HPP

#include <cmath>
#include <map>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include "soap/base/objectfactory.hpp"
#include "soap/options.hpp"
#include "soap/types.hpp"

namespace soap { namespace cgraph {
namespace ub = boost::numeric::ublas;
namespace bpy = boost::python;

typedef double dtype_t;
typedef ub::vector<dtype_t> vec_t;
typedef ub::matrix<dtype_t> mat_t;

struct CNode;
struct CNodeParams
{
    CNodeParams(int id, CNode *node);
    ~CNodeParams();
    CNode *getNode() { return node; }
    int getId() { return id; }
    int nParams() { return params.size(); }
    void setConstant(bool set_constant) { constant = set_constant; }
    bool isConstant() { return constant; }
    void setParamsNumpy(bpy::object &np_params, std::string np_dtype);
    void setParams(vec_t &arg_params) { params = arg_params; }
    // DATA
    int id;
    CNode *node;
    bool constant;
    bool active;
    vec_t params;
};

struct CGraphParams
{
    typedef std::vector<CNodeParams*> paramslist_t;
    CGraphParams() : id_counter(0) {;}
    ~CGraphParams();
    CNodeParams *allocate(CNode *node);
    int nParamSets() { return paramslist.size(); }
    int nParams();
    paramslist_t::iterator begin() { return paramslist.begin(); }
    paramslist_t::iterator end() { return paramslist.end(); }
    // DATA
    int id_counter;
    paramslist_t paramslist;
};

struct CNodeGrads
{
    typedef std::pair<int,mat_t*> paramset_t; // <- mat_t: [n_slots x n_params]
    typedef std::vector<paramset_t> sparse_grad_t;
    CNodeGrads() {;}
    ~CNodeGrads();
    sparse_grad_t::iterator begin() { return sparse_grads.begin(); }
    sparse_grad_t::iterator end() { return sparse_grads.end(); }
    mat_t *allocate(int params_id, int n_slots, int n_params);
    void listParamSets();
    void clear();
    void sort();
    void sumSlots();
    void add(CNodeGrads &other, vec_t &slot_scale);
    void add(CNodeGrads &other, dtype_t scale);
    bpy::object valsNumpy(int pid, std::string np_dtype);
    static void registerPython();
    // DATA
    std::map<int, bool> sparse_id_map; // NOTE Only used for sanity checks
    sparse_grad_t sparse_grads;
};

struct CGraph
{
    typedef std::vector<CNode*> nodelist_t;
    CGraph();
    ~CGraph();
    void clear();
    CNode *addInput();
    CNode *addTarget();
    CNode *addNodePython(std::string op, bpy::list &py_nodelist);
    CNode *addNode(std::string op, nodelist_t &nodelist);
    CNode *addOutputPython(std::string op, bpy::list &py_nodelist);
    CNode *addOutput(std::string op, nodelist_t &nodelist);
    CNode *addObjectivePython(std::string op, bpy::list &py_nodelist);
    CNode *addObjective(std::string op, nodelist_t &nodelist);
    nodelist_t::iterator beginNodes() { return nodes.begin(); }
    nodelist_t::iterator endNodes() { return nodes.end(); }
    nodelist_t::iterator beginDerived() { return derived.begin(); }
    nodelist_t::iterator endDerived() { return derived.end(); }
    nodelist_t::iterator beginObjectives() { return objectives.begin(); }
    nodelist_t::iterator endObjectives() { return objectives.end(); }
    nodelist_t::iterator beginOutputs() { return outputs.begin(); }
    nodelist_t::iterator endOutputs() { return outputs.end(); }
    nodelist_t::iterator beginTargets() { return targets.begin(); }
    nodelist_t::iterator endTargets() { return targets.end(); }
    CGraphParams::paramslist_t::iterator beginParams() { return params.begin(); }
    CGraphParams::paramslist_t::iterator endParams() { return params.end(); }
    CGraphParams &getParams() { return params; }
    int size() { return nodes.size(); }
    void allocateParams();
    void evaluateNumpy(bpy::object &npy_X, std::string np_dtype);
    void evaluate(mat_t &X);
    void feedNumpy(bpy::object &npy_X, bpy::object &npy_y, 
        std::string np_dtype_X, std::string np_dtype_Y);
    void feed(mat_t &X, mat_t &Y);
    static void registerPython();
    // DATA
    nodelist_t inputs;
    nodelist_t outputs;
    nodelist_t derived;
    nodelist_t nodes; // <- inputs + derived + outputs
    nodelist_t targets;
    nodelist_t objectives;
    CGraphParams params;
};

struct CNode
{
    CNode() : op("?"), active(true), params(NULL), params_constant(false) {;}
    virtual ~CNode();
    virtual void link(CGraph::nodelist_t &nodelist);
    virtual void resize(int n_slots);
    virtual int nParams() { return 0; }
    bpy::object valsNumpy(std::string np_dtype);
    CNodeGrads &getGrads() { return grads; }
    CNodeParams *getParams() { return params; }
    void setParamsConstant(bool set_constant);
    virtual void feed(mat_t &X, int colidx);
    virtual void evaluate();
    std::string getOp() { return op; }
    int inputSize() { return inputs.size(); }
    static void registerPython();
    // DATA
    std::string op;
    bool active;
    CNodeParams *params;
    bool params_constant;
    CGraph::nodelist_t inputs;
    vec_t vals;
    CNodeGrads grads;
};

struct CNodeIdentity : public CNode
{
    CNodeIdentity() { op = "identity"; }
    int nParams() { return 0; }
    void evaluate() { return; }
};

struct CNodeSigmoid : public CNode
{   
    CNodeSigmoid() { op = "sigmoid"; }
    int nParams() { return inputs.size()+1; }
    void evaluate();
};

struct CNodeLinear : public CNode
{   
    CNodeLinear() { op = "linear"; }
    int nParams() { return inputs.size()+1; }
    void evaluate();
};

struct CNodePearson : public CNode 
{
    CNodePearson() { op = "pearson"; }
    int nParams() { return 0; }
};

struct CNodeMSE : public CNode 
{
    CNodeMSE() { op = "mse"; }
    int nParams() { return 0; }
    void evaluate();
};

class CNodeFactory : public soap::base::ObjectFactory<std::string, CNode>
{
  private:
    CNodeFactory() {}
  public:
    static void registerAll(void);
    CNode *create(const std::string &key);
    friend CNodeFactory &CNodeCreator();
};

inline CNodeFactory &CNodeCreator() {
    static CNodeFactory _instance;
    return _instance;
}

inline CNode *CNodeFactory::create(const std::string &key) {
    assoc_map::const_iterator it(getObjects().find(key));
    if (it != getObjects().end()) {
        CNode *basis = (it->second)();
        return basis;
    } 
    else {
        throw std::runtime_error("Factory key '" + key + "' not found.");
    }
}

struct OptimizationAlgorithm
{
    OptimizationAlgorithm() : op("?") {;}
    virtual ~OptimizationAlgorithm() {;}
    virtual void fit(CGraph *cgraph, mat_t &X, mat_t &Y) = 0;
    std::string op;
};

struct AdaGrad : public OptimizationAlgorithm
{
    AdaGrad() { op = "adagrad"; }
    void fit(CGraph *cgraph, mat_t &X, mat_t &Y);
};

struct Optimizer
{
    Optimizer(Options *arg_options);
    ~Optimizer();
    void fitNumpy(CGraph *cgraph, bpy::object &np_X, bpy::object &np_Y,
        std::string np_dtype_X, std::string np_dtype_Y);
    static void registerPython();
    // DATA
    OptimizationAlgorithm *optalg;
    Options *options;
};

class OptimizerFactory : public soap::base::ObjectFactory<
    std::string, OptimizationAlgorithm>
{
  private:
    OptimizerFactory() {}
  public:
    static void registerAll(void);
    OptimizationAlgorithm *create(const std::string &key);
    friend OptimizerFactory &OptimizerCreator();
};

inline OptimizerFactory &OptimizerCreator() {
    static OptimizerFactory _instance;
    return _instance;
}

inline OptimizationAlgorithm *OptimizerFactory::create(const std::string &key) {
    assoc_map::const_iterator it(getObjects().find(key));
    if (it != getObjects().end()) {
        OptimizationAlgorithm *basis = (it->second)();
        return basis;
    } 
    else {
        throw std::runtime_error("Factory key '" + key + "' not found.");
    }
}


}}

#endif
