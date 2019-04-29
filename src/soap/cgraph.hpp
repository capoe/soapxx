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
#include "soap/base/rng.hpp"

namespace soap { namespace cgraph {
namespace ub = boost::numeric::ublas;
namespace bpy = boost::python;

typedef double dtype_t;
typedef ub::vector<dtype_t> vec_t;
typedef ub::matrix<dtype_t> mat_t;

struct CNode;
struct CNodeGrads;
struct CNodeParams
{
    CNodeParams();
    CNodeParams(int id, CNode *node);
    ~CNodeParams();
    CNode *getNode() { return node; }
    int getId() { return id; }
    int nParams() { return params.size(); }
    void setConstant(bool set_constant) { constant = set_constant; }
    CNodeParams *deepCopy();
    bool isConstant() { return constant; }
    bpy::object valsNumpy(std::string np_dtype);
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
    CGraphParams *deepCopy();
    void add(CNodeGrads &grads, dtype_t coeff);
    void addSquare(CNodeGrads &grads, dtype_t coeff1, dtype_t coeff2);
    void add(CNodeGrads &grads, dtype_t coeff, CGraphParams &frictions);
    int nParamSets() { return paramslist.size(); }
    int nParams();
    paramslist_t::iterator begin() { return paramslist.begin(); }
    paramslist_t::iterator end() { return paramslist.end(); }
    // DATA
    int id_counter;
    paramslist_t paramslist;
};

class CNodeGrads
{
  public:
    typedef std::pair<int,mat_t*> paramset_t; // <- mat_t: [n_params x n_slots]
    typedef std::vector<paramset_t> sparse_grad_t;
    CNodeGrads() {;}
    ~CNodeGrads();
    sparse_grad_t::iterator begin() { return sparse_grads.begin(); }
    sparse_grad_t::iterator end() { return sparse_grads.end(); }
    mat_t *allocate(int params_id, int n_slots, int n_params);
    void listParamSets();
    void clear();
    void sort();
    int size() { return sparse_grads.size(); }
    void sumSlots();
    void add(CNodeGrads &other, vec_t &slot_scale);
    void add(CNodeGrads &other, dtype_t scale);
    bpy::object valsNumpy(int pid, std::string np_dtype);
    static void registerPython();
  private:
    // DATA
    std::map<int, bool> sparse_id_map; // NOTE Only used for sanity checks
    sparse_grad_t sparse_grads;
};

class CGraph
{
  public:
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
    nodelist_t::iterator beginOutputs() { return outputs.begin(); }
    nodelist_t::iterator endOutputs() { return outputs.end(); }
    nodelist_t::iterator beginTargets() { return targets.begin(); }
    nodelist_t::iterator endTargets() { return targets.end(); }
    nodelist_t::iterator beginObjectives() { return objectives.begin(); }
    nodelist_t::iterator endObjectives() { return objectives.end(); }
    nodelist_t &getNodes() { return nodes; }
    nodelist_t &getInputs() { return inputs; }
    nodelist_t &getDerived() { return derived; }
    nodelist_t &getOutputs() { return outputs; }
    nodelist_t &getTargets() { return targets; }
    nodelist_t &getObjectives() { return objectives; }
    CGraphParams::paramslist_t::iterator beginParams() { return params.begin(); }
    CGraphParams::paramslist_t::iterator endParams() { return params.end(); }
    CGraphParams &getParams() { return params; }
    int size() { return nodes.size(); }
    void bypassInactiveNodes();
    void allocateParams();
    void allocateParamsFor(CNode *node);
    void evaluateNumpy(bpy::object &npy_X, std::string np_dtype);
    void evaluateInputGradsNumpy(bpy::object &npy_X, std::string np_dtype);
    void evaluate(mat_t &X, bool with_input_grads);
    void feedNumpy(bpy::object &npy_X, bpy::object &npy_y, 
        std::string np_dtype_X, std::string np_dtype_Y);
    void feed(mat_t &X, mat_t &Y, bool with_input_grads);
    static void registerPython();
  private:
    CNode *createBlank(std::string op);
    // DATA
    int id_counter;
    nodelist_t inputs;
    nodelist_t outputs;
    nodelist_t derived;
    nodelist_t nodes; // <- inputs + derived + outputs
    nodelist_t targets;
    nodelist_t objectives;
    CGraphParams params;
};

struct CNodeDropout
{
    CNodeDropout(int seed);
    ~CNodeDropout() { if (rng) delete rng; }
    void affect(bpy::list &py_nodelist, 
        bpy::object &np_probs, std::string np_dtype);
    void sample();
    static void registerPython();
    // DATA
    soap::base::RNG *rng;
    vec_t probs_active;
    CGraph::nodelist_t nodelist;
};

struct CNode
{
    CNode();
    virtual ~CNode();
    virtual void linkPython(bpy::list &py_nodelist);
    virtual void link(CGraph::nodelist_t &nodelist);
    virtual void resize(int n_slots);
    virtual int nParams() { return 0; }
    bpy::object valsNumpy(std::string np_dtype);
    CNodeGrads &getGrads() { return grads; }
    CNodeGrads &getInputGrads() { return input_grads; }
    CNodeParams *getParams() { return params; }
    bpy::list getInputsPython();
    void setId(int arg_id) { id = arg_id; }
    void setTag(std::string arg_tag) { tag = arg_tag; }
    std::string getTag() { return tag; }
    void setParamsConstant(bool set_constant);
    bool isParamsConstant() { return params_constant; }
    void setBranchActive(bool set_active);
    bool isActive() { return active; }
    void setActive(bool set_active) { active = set_active; }
    virtual void zero();
    virtual void resetAndResize();
    virtual void feed(mat_t &X, int colidx, bool with_input_grads);
    virtual void evaluate(bool with_input_grads);
    std::string getOp() { return op; }
    int inputSize() { return inputs.size(); }
    static void registerPython();
    // DATA
    int id;
    std::string tag;
    std::string op;
    bool active;
    CNodeParams *params;
    bool params_constant;
    CGraph::nodelist_t inputs;
    vec_t vals;
    CNodeGrads grads;
    CNodeGrads input_grads;
};

struct CNodeInput : public CNode
{
    CNodeInput() { op = "input"; }
    int nParams() { return 0; }
    void evaluate(bool with_input_grads);
};

struct CNodeSigmoid : public CNode
{   
    CNodeSigmoid() { op = "sigmoid"; }
    int nParams() { return inputs.size()+1; }
    void evaluate(bool with_input_grads);
};

struct CNodeLinear : public CNode
{   
    CNodeLinear() { op = "linear"; }
    int nParams() { return inputs.size()+1; }
    void evaluate(bool with_input_grads);
};

struct CNodeExp : public CNode
{
    CNodeExp() { op = "exp"; }
    int nParams() { return 1; }
    void evaluate(bool with_input_grads);
};

struct CNodeLog : public CNode
{
    CNodeLog() { op = "log"; }
    int nParams() { return 0; }
    void evaluate(bool with_input_grads);
};

struct CNodeMod : public CNode
{
    CNodeMod() { op = "mod"; }
    int nParams() { return 0; }
    void evaluate(bool with_input_grads);
};

struct CNodePow : public CNode
{
    CNodePow() { op = "pow"; }
    int nParams() { return 1; }
    void evaluate(bool with_input_grads);
};

struct CNodeMult : public CNode
{
    CNodeMult() { op = "mult"; }
    int nParams() { return 0; }
    void evaluate(bool with_input_grads);
};

struct CNodeDiv : public CNode
{
    CNodeDiv() { op = "div"; }
    int nParams() { return 0; }
    void evaluate(bool with_input_grads);
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
    void evaluate(bool with_input_grads);
};

struct CNodeXENT : public CNode 
{
    CNodeXENT() { op = "xent"; }
    int nParams() { return 0; }
    void evaluate(bool with_input_grads);
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
    virtual void step(CGraph *cgraph, mat_t &X, mat_t &Y, 
        int n_steps, double rate) = 0;
    std::string op;
};

struct OptAdaGrad : public OptimizationAlgorithm
{
    OptAdaGrad() { op = "adagrad"; frictions = NULL; }
    ~OptAdaGrad();
    void fit(CGraph *cgraph, mat_t &X, mat_t &Y);
    virtual void step(CGraph *cgraph, mat_t &X, mat_t &Y, 
        int n_steps, double rate);
    CGraphParams *frictions;
};

struct OptSteep : public OptimizationAlgorithm
{
    OptSteep() { op = "steep"; }
    void fit(CGraph *cgraph, mat_t &X, mat_t &Y);
    virtual void step(CGraph *cgraph, mat_t &X, mat_t &Y, 
        int n_steps, double rate);
};

struct Optimizer
{
    Optimizer();
    Optimizer(std::string method);
    Optimizer(Options *arg_options);
    ~Optimizer();
    void fitNumpy(CGraph *cgraph, bpy::object &np_X, bpy::object &np_Y,
        std::string np_dtype_X, std::string np_dtype_Y);
    void stepNumpy(CGraph *cgraph, bpy::object &np_X, bpy::object &np_Y,
        std::string np_dtype_X, std::string np_dtype_Y,
        int n_steps, double rate);
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
