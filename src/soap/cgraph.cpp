#include <algorithm>
#include <assert.h>
#include <fstream>
#include <numeric>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "soap/base/tokenizer.hpp"
#include "soap/base/exceptions.hpp"
#include "soap/cgraph.hpp"
#include "soap/globals.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"
#include "boost/format.hpp"

namespace soap { namespace cgraph {

CGraph::CGraph() {
    ;
}

CGraph::~CGraph() {
    this->clear();
}

void CGraph::clear() {
    for (auto it=nodes.begin(); it!=nodes.end(); ++it) {
        delete *it;
    }
    for (auto it=objectives.begin(); it!=objectives.end(); ++it) {
        delete *it;
    }
    for (auto it=targets.begin(); it!=targets.end(); ++it) {
        delete *it;
    }
    inputs.clear();
    outputs.clear();
    targets.clear();
    objectives.clear();
    nodes.clear();
    derived.clear();
}

CNode *CGraph::addInput() {
    CNode *new_node = CNodeCreator().create("identity");
    inputs.push_back(new_node);
    nodes.push_back(new_node);
    return new_node;
}

CNode *CGraph::addTarget() {
    CNode *new_node = CNodeCreator().create("identity");
    targets.push_back(new_node);
    return new_node;
}

CNode *CGraph::addNodePython(std::string op, boost::python::list &py_nodelist) {
    nodelist_t nodelist;
    for (int i=0; i<boost::python::len(py_nodelist); ++i) {
        nodelist.push_back(boost::python::extract<CNode*>(py_nodelist[i]));
    }
    CNode *new_node = this->addNode(op, nodelist);
    return new_node;
}

CNode *CGraph::addNode(std::string op, CGraph::nodelist_t &nodelist) {
    CNode *new_node = CNodeCreator().create(op);
    new_node->link(nodelist);
    nodes.push_back(new_node);
    derived.push_back(new_node);
    return new_node;
}

CNode *CGraph::addOutputPython(std::string op, boost::python::list &py_nodelist) {
    nodelist_t nodelist;
    for (int i=0; i<boost::python::len(py_nodelist); ++i) {
        nodelist.push_back(boost::python::extract<CNode*>(py_nodelist[i]));
    }
    CNode *new_node = this->addOutput(op, nodelist);
    return new_node;
}

CNode *CGraph::addOutput(std::string op, CGraph::nodelist_t &nodelist) {
    CNode *new_node = CNodeCreator().create(op);
    new_node->link(nodelist);
    nodes.push_back(new_node);
    derived.push_back(new_node);
    outputs.push_back(new_node);
    return new_node;
}

CNode *CGraph::addObjectivePython(std::string op, boost::python::list &py_nodelist) {
    nodelist_t nodelist;
    for (int i=0; i<boost::python::len(py_nodelist); ++i) {
        nodelist.push_back(boost::python::extract<CNode*>(py_nodelist[i]));
    }
    CNode *new_node = this->addObjective(op, nodelist);
    return new_node;
}

CNode *CGraph::addObjective(std::string op, CGraph::nodelist_t &nodelist) {
    CNode *new_node = CNodeCreator().create(op);
    new_node->link(nodelist);
    objectives.push_back(new_node);
    return new_node;
}

void CGraph::allocateParams() {
    for (auto it=beginNodes(); it!=endNodes(); ++it) {
        params.allocate(*it);
    }
    for (auto it=beginObjectives(); it!=endObjectives(); ++it) {
        params.allocate(*it);
    }
    GLOG() << "Allocated " << params.nParams() << " parameters across " 
        << params.nParamSets() << " sets" << std::endl;
}

void CGraph::evaluateNumpy(bpy::object &np_X, std::string np_dtype_X) {
    mat_t X;
    soap::linalg::numpy_converter npc_X(np_dtype_X.c_str());
    npc_X.numpy_to_ublas<dtype_t>(np_X, X);
    this->evaluate(X);
}

void CGraph::evaluate(mat_t &X) {
    assert(X.size2() == inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        inputs[i]->feed(X, i);
    }
    int i = 0;
    for (auto it=beginDerived(); it!=endDerived(); ++it, ++i) {
        GLOG() << "\rEvaluating derived node " << i << std::flush;
        (*it)->evaluate();
    }
    GLOG() << std::endl;
}

void CGraph::feedNumpy(bpy::object &np_X, bpy::object &np_Y, 
        std::string np_dtype_X, std::string np_dtype_Y) {
    mat_t X;
    soap::linalg::numpy_converter npc_X(np_dtype_X.c_str());
    npc_X.numpy_to_ublas<dtype_t>(np_X, X);
    mat_t Y;
    soap::linalg::numpy_converter npc_Y(np_dtype_Y.c_str());
    npc_Y.numpy_to_ublas<dtype_t>(np_Y, Y);
    this->feed(X, Y);
}

void CGraph::feed(mat_t &X, mat_t &Y) {
    GLOG() << "Feeding CGraph with X:[" << X.size1() << "x" << X.size2() << "], Y:[" 
        << Y.size1() << "x" << Y.size2() << "]" << std::endl;
    assert(X.size2() == inputs.size());
    assert(Y.size2() == targets.size());
    // Initialize root (input) nodes
    for (int i=0; i<inputs.size(); ++i) {
        inputs[i]->feed(X, i);
    }
    for (int i=0; i<targets.size(); ++i) {
        targets[i]->feed(Y, i);
    }
    // Evaluate derived nodes
    int i = 0;
    for (auto it=beginDerived(); it!=endDerived(); ++it, ++i) {
        GLOG() << "\rEvaluating derived node " << i << std::flush;
        (*it)->evaluate();
    }
    GLOG() << std::endl;
    // Evaluate objectives
    int j = 0;
    for (auto it=beginObjectives(); it!=endObjectives(); ++it, ++j) {
        GLOG() << "\rEvaluating objective node " << j << std::flush;
        (*it)->evaluate();
    }
    GLOG() << std::endl;
}

void CGraph::registerPython() {
    using namespace boost::python;
    class_<CGraph, CGraph*>("CGraph", init<>())
        .add_property("size", &CGraph::size)
        .add_property("nodes", 
            range<return_value_policy<reference_existing_object>>(
            &CGraph::beginNodes, &CGraph::endNodes))
        .add_property("objectives", 
            range<return_value_policy<reference_existing_object>>(
            &CGraph::beginObjectives, &CGraph::endObjectives))
        .add_property("outputs", 
            range<return_value_policy<reference_existing_object>>(
            &CGraph::beginOutputs, &CGraph::endOutputs))
        .add_property("params", 
            range<return_value_policy<reference_existing_object>>(
            &CGraph::beginParams, &CGraph::endParams))
        .def("__len__", &CGraph::size)
        .def("allocateParams", &CGraph::allocateParams)
        .def("evaluate", &CGraph::evaluateNumpy)
        .def("feed", &CGraph::feedNumpy)
        .def("addInput", &CGraph::addInput, 
            return_value_policy<reference_existing_object>())
        .def("addTarget", &CGraph::addTarget, 
            return_value_policy<reference_existing_object>())
        .def("addNode", &CGraph::addNodePython, 
            return_value_policy<reference_existing_object>())
        .def("addOutput", &CGraph::addOutputPython, 
            return_value_policy<reference_existing_object>())
        .def("addObjective", &CGraph::addObjectivePython, 
            return_value_policy<reference_existing_object>());
    class_<CNodeParams, CNodeParams*>("CNodeParams", no_init)
        .add_property("size", &CNodeParams::nParams)
        .add_property("id", &CNodeParams::getId)
        .def("set", &CNodeParams::setParamsNumpy)
        .def("node", &CNodeParams::getNode, 
            return_value_policy<reference_existing_object>());
    class_<CGraph::nodelist_t>("NodeList")
        .def(vector_indexing_suite<CGraph::nodelist_t>());
    class_<CGraphParams::paramslist_t>("ParamsList")
        .def(vector_indexing_suite<CGraphParams::paramslist_t>());
}

// ===========
// CNODEPARAMS
// ===========

CNodeParams::CNodeParams(int params_id, CNode *arg_node)
    : id(params_id), constant(false), active(true), node(arg_node)  {
    params = vec_t(node->nParams(), 0.);
    constant = arg_node->params_constant;
    arg_node->params = this;
}

CNodeParams::~CNodeParams() {
    ;
}

void CNodeParams::setParamsNumpy(bpy::object &np_params, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    npc.numpy_to_ublas<dtype_t>(np_params, params);
}

// ============
// CGRAPHPARAMS
// ============

CGraphParams::~CGraphParams() {
    for (auto it=begin(); it!=end(); ++it) {
        delete (*it);
    }
    paramslist.clear();
}

int CGraphParams::nParams() {
    int n = 0;
    for (auto it=begin(); it!=end(); ++it) {
        n += (*it)->params.size();
    }
    return n;
}

CNodeParams *CGraphParams::allocate(CNode *node) {
    assert(node->params == NULL && 
        "Overwriting existing parameter set not permitted");
    id_counter += 1;
    CNodeParams *new_params = new CNodeParams(id_counter, node);
    paramslist.push_back(new_params);
    return new_params;
}

// =====
// CNODE
// =====

CNode::~CNode() {
    ;
}

void CNode::setParamsConstant(bool set_constant) {
    params_constant = set_constant;
    if (params != NULL) params->setConstant(set_constant);
}

void CNode::resize(int n_slots) {
    bool preserve = false;
    vals.resize(n_slots, preserve);
    //vals = vec_t(n_slots, 0.);
}

void CNode::feed(mat_t &X, int colidx) {
    if (vals.size() != X.size1()) this->resize(X.size1());
    for (int i=0; i<X.size1(); ++i) {
        vals(i) = X(i,colidx);
    }
}

bpy::object CNode::valsNumpy(std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    return npc.ublas_to_numpy<dtype_t>(vals);
}

void CNode::evaluate() {
    assert(false);
}

void CNode::link(CGraph::nodelist_t &nodelist) {
    for (auto it=nodelist.begin(); it!=nodelist.end(); ++it) 
        inputs.push_back(*it);
}

void CNode::registerPython() {
    using namespace boost::python;
    class_<CNode, CNode*>("CNode", init<>())
        .def("vals", &CNode::valsNumpy)
        .def("grads", &CNode::getGrads, 
            return_value_policy<reference_existing_object>())
        .def("params", &CNode::getParams, 
            return_value_policy<reference_existing_object>())
        .def("setParamsConstant", &CNode::setParamsConstant)
        .add_property("size", &CNode::inputSize)
        .add_property("op", &CNode::getOp);
    class_<CNodeGrads, CNodeGrads*>("CNodeGrads", init<>())
        .def("vals", &CNodeGrads::valsNumpy)
        .def("listParamSets", &CNodeGrads::listParamSets);
    class_<Optimizer, Optimizer*>("Optimizer", init<Options*>()) 
        .def("fit", &Optimizer::fitNumpy);
}

void CNodeSigmoid::evaluate() {
    vec_t &p = params->params;
    // Allocate output
    int n_slots = vals.size();
    if (inputs.size() > 0) {
        n_slots = inputs[0]->vals.size();
        this->resize(n_slots);
    }
    int n_params = p.size();
    int n_inputs = inputs.size();
    // Evaluate output
    for (int a=0; a<n_slots; ++a) {
        vals(a) = -p(n_inputs); // Last parameter is bias
    }
    for (int i=0; i<n_inputs; ++i) {
        vec_t &vi = inputs[i]->vals;
        for (int a=0; a<n_slots; ++a) {
            vals(a) -= vi(a)*p(i);
        }
    }
    vec_t slot_scale = vec_t(n_slots, 0.);
    for (int a=0; a<n_slots; ++a) {
        vals(a) = 1./(1. + std::exp(vals(a)));
        slot_scale(a) = vals(a)*(1-vals(a));
    }
    // Evaluate gradients
    grads.clear();
    if (!params->isConstant()) {
        mat_t &pgrads = *(grads.allocate(params->id, n_slots, n_params));
        for (int i=0; i<n_inputs; ++i) {
            vec_t &vi = inputs[i]->vals;
            for (int a=0; a<n_slots; ++a) {
                pgrads(a,i) = slot_scale(a)*vi(a);
            }
        }
        for (int a=0; a<n_slots; ++a) {
            pgrads(a,n_inputs) = slot_scale(a)*1.; // Last parameter is bias
        }
    }
    for (int i=0; i<n_inputs; ++i) {
        vec_t grad_slot_scale = p(i)*slot_scale;
        grads.add(inputs[i]->grads, grad_slot_scale);
    }
}

void CNodeLinear::evaluate() {
    vec_t &p = params->params;
    // Allocate output
    int n_slots = vals.size();
    if (inputs.size() > 0) {
        n_slots = inputs[0]->vals.size();
        this->resize(n_slots);
    }
    int n_params = p.size();
    int n_inputs = inputs.size();
    // Evaluate output
    for (int a=0; a<vals.size(); ++a) {
        vals(a) = p(inputs.size()); // Last parameter is y-intercept
    }
    for (int i=0; i<inputs.size(); ++i) {
        vec_t &vi = inputs[i]->vals;
        for (int a=0; a<vals.size(); ++a) {
            vals(a) += vi(a)*p(i);
        }
    }
    // Evaluate gradients
    grads.clear();
    if (!params->isConstant()) {
        mat_t &pgrads = *(grads.allocate(params->id, vals.size(), p.size()));
        for (int i=0; i<inputs.size(); ++i) {
            vec_t &vi = inputs[i]->vals;
            for (int a=0; a<vals.size(); ++a) {
                pgrads(a,i) = vi(a);
            }
        }
        for (int a=0; a<vals.size(); ++a) {
            pgrads(a,inputs.size()) = 1.; // Last parameter is bias
        }
    }
    for (int i=0; i<n_inputs; ++i) {
        dtype_t grad_scale = p(i);
        grads.add(inputs[i]->grads, grad_scale);
    }
}

void CNodeMSE::evaluate() {
    int n_inputs = inputs.size();
    assert(n_inputs == 2 && "MSE node only allows two inputs = [output,target]");
    int n_slots = inputs[0]->vals.size();
    assert(inputs[1]->vals.size() == n_slots && "MSE: Input vector size mismatch");
    // Allocate output
    this->resize(1);
    vals(0) = 0.;
    vec_t grad_slot_scale(n_slots, 0.);
    // Evaluate output
    vec_t &y0 = inputs[0]->vals;
    vec_t &y1 = inputs[1]->vals;
    for (int a=0; a<n_slots; ++a) {
        vals(0) += std::pow(y0(a)-y1(a),2);
        grad_slot_scale(a) = y0(a)-y1(a);
    }
    vals *= 1./n_slots;
    grad_slot_scale *= 2./n_slots;
    // Evaluate gradients
    grads.clear();
    grads.add(inputs[0]->grads, grad_slot_scale);
    grads.sumSlots();
}

// ==========
// CNODEGRADS
// ==========

CNodeGrads::~CNodeGrads() {
    this->clear();
}

void CNodeGrads::clear() {
    for (auto it=begin(); it!=end(); ++it) {
        delete it->second;
    }
    sparse_grads.clear();
    sparse_id_map.clear();
}

void CNodeGrads::sort() {
    std::sort(begin(), end(), 
        [&](paramset_t g1, paramset_t g2) {
            return g1.first <= g2.first;
        }
    );
}

void CNodeGrads::sumSlots() {
    for (auto it=begin(); it!=end(); ++it) {
        mat_t *full = it->second;
        mat_t *reduced = new mat_t(1, full->size2(), 0.);
        for (int i=0; i<full->size1(); ++i) {
            for (int j=0; j<full->size2(); ++j) {
                (*reduced)(0,j) += (*full)(i,j);
            }
        }
        delete full;
        it->second = reduced;
    }
}

void CNodeGrads::add(CNodeGrads &other, vec_t &slot_scale) {
    sparse_grad_t add_entries; 
    auto it = this->begin();
    auto jt = other.begin();
    while (jt != other.end()) {
        if (it == this->end() || jt->first < it->first) {
            mat_t *add_mat = new mat_t(jt->second->size1(), jt->second->size2());
            // TODO Accelerate >>>
            for (int i=0; i<jt->second->size1(); ++i) {
                for (int j=0; j<jt->second->size2(); ++j) {
                    (*add_mat)(i,j) = slot_scale(i)*(*(jt->second))(i,j);
                }
            }
            // <<< Accelerate TODO
            add_entries.push_back(paramset_t(jt->first, add_mat));
            ++jt;
        } else if (jt->first == it->first) {
            // TODO Accelerate >>>
            for (int i=0; i<jt->second->size1(); ++i) {
                for (int j=0; j<jt->second->size2(); ++j) {
                    (*(it->second))(i,j) += slot_scale(i)*(*(jt->second))(i,j);
                }
            }
            // <<< Accelerate TODO
            ++it;
            ++jt;
        } else if (jt->first > it->first) {
            ++it;
        }
    }
    for (auto it=add_entries.begin(); it!=add_entries.end(); ++it) {
        sparse_grads.push_back(*it);
    }
    this->sort();
}

void CNodeGrads::add(CNodeGrads &other, dtype_t scale) {
    sparse_grad_t add_entries; 
    auto it = this->begin();
    auto jt = other.begin();
    while (jt != other.end()) {
        if (it == this->end() || jt->first < it->first) {
            mat_t *add_mat = new mat_t(jt->second->size1(), jt->second->size2());
            *add_mat = scale*(*(jt->second));
            add_entries.push_back(paramset_t(jt->first, add_mat));
            ++jt;
        } else if (jt->first == it->first) {
            (*(it->second)) += scale*(*(jt->second));
            ++it;
            ++jt;
        } else if (jt->first > it->first) {
            ++it;
        }
    }
    for (auto it=add_entries.begin(); it!=add_entries.end(); ++it) {
        sparse_grads.push_back(*it);
    }
    this->sort();
}

mat_t *CNodeGrads::allocate(int params_id, int n_slots, int n_params) {
    assert(sparse_id_map.find(params_id) == sparse_id_map.end());
    mat_t *new_mat = new mat_t(n_slots, n_params, 0.);   
    sparse_grads.push_back(paramset_t(params_id, new_mat));
    sparse_id_map[params_id] = true;
    this->sort();
    return new_mat;
}

void CNodeGrads::listParamSets() {
    GLOG << "Dependent on " << sparse_grads.size() << " parameter sets" << std::endl;
    for (auto it=begin(); it!=end(); ++it) {
        GLOG() << " - PID=" << it->first << " Size=[" << it->second->size1()
            << "x" << it->second->size2() << "]" << std::endl;
    }
}

bpy::object CNodeGrads::valsNumpy(int pid, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    for (auto it=begin(); it!=end(); ++it) {
        if (it->first == pid) {
            return npc.ublas_to_numpy<dtype_t>(*(it->second));
        }
    }
    throw soap::base::OutOfRange("PID out of range");
    return bpy::object();
}

// ============
// OPTIMIZATION
// ============

Optimizer::Optimizer(Options *arg_options) : options(arg_options) {
    optalg = OptimizerCreator().create(options->get<std::string>("opt.type"));
}

Optimizer::~Optimizer() {
    ;
}

void Optimizer::fitNumpy(CGraph *cgraph, bpy::object &np_X, bpy::object &np_Y,
        std::string np_dtype_X, std::string np_dtype_Y) {
    mat_t X;
    soap::linalg::numpy_converter npc_X(np_dtype_X.c_str());
    npc_X.numpy_to_ublas<dtype_t>(np_X, X);
    mat_t Y;
    soap::linalg::numpy_converter npc_Y(np_dtype_Y.c_str());
    npc_Y.numpy_to_ublas<dtype_t>(np_Y, Y);
    optalg->fit(cgraph, X, Y);
}

void AdaGrad::fit(CGraph *cgraph, mat_t &X, mat_t &Y) {
    GLOG() << "Starting " << op << " optimization" << std::endl;
    cgraph->feed(X, Y);
    CGraphParams &params = cgraph->getParams();
    GLOG() << "CGraph with " << cgraph->size() << " nodes, " 
        << params.nParamSets() << " parameter sets" << std::endl;
    for (auto it=cgraph->beginObjectives(); it!=cgraph->endObjectives(); ++it) {
        CNode *eps = *it;
        CNodeGrads &grads = eps->getGrads();
        GLOG() << "Objective node with " << grads.sparse_grads.size() 
            << " gradient sets" << std::endl;
    }
}

void CNodeFactory::registerAll(void) {
	CNodeCreator().Register<CNodeIdentity>("identity");
	CNodeCreator().Register<CNodeLinear>("linear");
	CNodeCreator().Register<CNodeSigmoid>("sigmoid");
	CNodeCreator().Register<CNodePearson>("pearson");
	CNodeCreator().Register<CNodeMSE>("mse");
}

void OptimizerFactory::registerAll(void) {
    OptimizerCreator().Register<AdaGrad>("adagrad");
}

}}

