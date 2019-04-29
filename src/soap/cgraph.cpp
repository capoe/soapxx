#include <algorithm>
#include <assert.h>
#include <fstream>
#include <numeric>
#include <boost/format.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "soap/base/tokenizer.hpp"
#include "soap/base/exceptions.hpp"
#include "soap/cgraph.hpp"
#include "soap/globals.hpp"
#include "soap/linalg/numpy.hpp"
#include "soap/linalg/operations.hpp"

namespace soap { namespace cgraph {

CGraph::CGraph() : id_counter(0) {
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

CNode *CGraph::createBlank(std::string op) {
    id_counter += 1;
    CNode *new_node = CNodeCreator().create(op);
    new_node->setId(id_counter);
    return new_node;
}

CNode *CGraph::addInput() {
    CNode *new_node = createBlank("input");
    allocateParamsFor(new_node);
    inputs.push_back(new_node);
    nodes.push_back(new_node);
    return new_node;
}

CNode *CGraph::addTarget() {
    CNode *new_node = createBlank("input");
    allocateParamsFor(new_node);
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
    CNode *new_node = createBlank(op);
    new_node->link(nodelist);
    allocateParamsFor(new_node);
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
    CNode *new_node = createBlank(op);
    new_node->link(nodelist);
    allocateParamsFor(new_node);
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
    CNode *new_node = createBlank(op);
    new_node->link(nodelist);
    allocateParamsFor(new_node);
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

void CGraph::allocateParamsFor(CNode *node) {
    params.allocate(node);
}

void CGraph::evaluateNumpy(bpy::object &np_X, std::string np_dtype_X) {
    mat_t X;
    soap::linalg::numpy_converter npc_X(np_dtype_X.c_str());
    npc_X.numpy_to_ublas<dtype_t>(np_X, X);
    bool with_input_grads = false;
    this->evaluate(X, with_input_grads);
}

void CGraph::evaluateInputGradsNumpy(bpy::object &np_X, std::string np_dtype_X) {
    mat_t X;
    soap::linalg::numpy_converter npc_X(np_dtype_X.c_str());
    npc_X.numpy_to_ublas<dtype_t>(np_X, X);
    bool with_input_grads = true;
    this->evaluate(X, with_input_grads);
}

void CGraph::evaluate(mat_t &X, bool with_input_grads) {
    assert(X.size2() == inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        inputs[i]->feed(X, i, with_input_grads);
    }
    int i = 0;
    for (auto it=beginDerived(); it!=endDerived(); ++it, ++i) {
        GLOG() << "\rEvaluating derived node " << i << std::flush;
        (*it)->evaluate(with_input_grads);
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
    bool with_input_grads = false;
    this->feed(X, Y, with_input_grads);
}

void CGraph::feed(mat_t &X, mat_t &Y, bool with_input_grads) {
    GLOG() << "Feeding CGraph with X:[" << X.size1() << "x" << X.size2() << "], Y:[" 
        << Y.size1() << "x" << Y.size2() << "]" << std::endl;
    assert(X.size2() == inputs.size());
    assert(Y.size2() == targets.size());
    // Initialize root (input) nodes
    for (int i=0; i<inputs.size(); ++i) {
        inputs[i]->feed(X, i, with_input_grads);
    }
    for (int i=0; i<targets.size(); ++i) {
        targets[i]->feed(Y, i, false);
    }
    // Evaluate derived nodes
    int i = 0;
    for (auto it=beginDerived(); it!=endDerived(); ++it, ++i) {
        GLOG() << "\rEvaluating derived node " << i << std::flush;
        (*it)->evaluate(with_input_grads);
    }
    GLOG() << std::endl;
    // Evaluate objectives
    int j = 0;
    for (auto it=beginObjectives(); it!=endObjectives(); ++it, ++j) {
        GLOG() << "\rEvaluating objective node " << j << std::flush;
        (*it)->evaluate(with_input_grads);
    }
    GLOG() << std::endl;
}

void CGraph::bypassInactiveNodes() {
    nodelist_t derived_out;
    for (auto it=beginDerived(); it!=endDerived(); ++it) {
        if ((*it)->isActive()) derived_out.push_back(*it);
        else ;
    }
    derived = derived_out;
}

void CGraph::registerPython() {
    using namespace boost::python;
    class_<CGraph, CGraph*>("CGraph", init<>())
        .add_property("size", &CGraph::size)
        .add_property("params", 
            range<return_value_policy<reference_existing_object>>(
            &CGraph::beginParams, &CGraph::endParams))
        .def("nodes", &CGraph::getNodes, 
            return_value_policy<reference_existing_object>())
        .def("inputs", &CGraph::getInputs, 
            return_value_policy<reference_existing_object>())
        .def("derived", &CGraph::getDerived, 
            return_value_policy<reference_existing_object>())
        .def("outputs", &CGraph::getOutputs, 
            return_value_policy<reference_existing_object>())
        .def("targets", &CGraph::getTargets, 
            return_value_policy<reference_existing_object>())
        .def("objectives", &CGraph::getObjectives, 
            return_value_policy<reference_existing_object>())
        .def("__len__", &CGraph::size)
        .def("bypassInactiveNodes", &CGraph::bypassInactiveNodes)
        .def("allocateParams", &CGraph::allocateParams)
        .def("allocateParamsFor", &CGraph::allocateParamsFor)
        .def("evaluate", &CGraph::evaluateNumpy)
        .def("evaluateInputGrads", &CGraph::evaluateInputGradsNumpy)
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
        .def("vals", &CNodeParams::valsNumpy)
        .def("set", &CNodeParams::setParamsNumpy)
        .def("node", &CNodeParams::getNode, 
            return_value_policy<reference_existing_object>());
    class_<CGraph::nodelist_t>("NodeList")
        .def(vector_indexing_suite<CGraph::nodelist_t>());
    class_<CGraphParams::paramslist_t>("ParamsList")
        .def(vector_indexing_suite<CGraphParams::paramslist_t>());
    class_<CNodeDropout, CNodeDropout*>("CNodeDropout", init<int>())
        .def("affect", &CNodeDropout::affect)
        .def("sample", &CNodeDropout::sample);
}

CNodeDropout::CNodeDropout(int seed) {
    rng = new soap::base::RNG();
    srand(seed);
    rng->init(rand(), rand(), rand(), rand());
}

void CNodeDropout::affect(bpy::list &py_nodelist, 
        bpy::object &np_probs, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    npc.numpy_to_ublas<dtype_t>(np_probs, probs_active);
    nodelist.clear();
    for (int i=0; i<boost::python::len(py_nodelist); ++i) {
        nodelist.push_back(boost::python::extract<CNode*>(py_nodelist[i]));
    }
    assert(probs_active.size() == nodelist.size() 
        && "Inconsistent arg dimensions in CNodeDropout::affect");
}

void CNodeDropout::sample() {
    int i = 0;
    for (auto it=nodelist.begin(); it!=nodelist.end(); ++it, ++i) {
        double p = rng->uniform();
        if (p <= probs_active[i]) {
            (*it)->setActive(true);
        } else {
            (*it)->setActive(false);
        }
    }
}

// ===========
// CNODEPARAMS
// ===========

CNodeParams::CNodeParams()
    : id(-1), node(NULL), constant(false), active(false) {
}

CNodeParams::CNodeParams(int params_id, CNode *arg_node)
    : id(params_id), constant(false), active(true), node(arg_node)  {
    params = vec_t(node->nParams(), 0.);
    constant = arg_node->params_constant;
    arg_node->params = this;
}

CNodeParams *CNodeParams::deepCopy() {
    CNodeParams *cnparams = new CNodeParams();
    cnparams->id = id;
    cnparams->node = node;
    cnparams->constant = constant;
    cnparams->active = active;
    cnparams->params = vec_t(params.size(), 0.);
    return cnparams;
}

CNodeParams::~CNodeParams() {
    ;
}

void CNodeParams::setParamsNumpy(bpy::object &np_params, std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    npc.numpy_to_ublas<dtype_t>(np_params, params);
}

bpy::object CNodeParams::valsNumpy(std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    return npc.ublas_to_numpy<dtype_t>(params);
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

CGraphParams *CGraphParams::deepCopy() {
    CGraphParams *cgparams = new CGraphParams();
    cgparams->id_counter = id_counter;
    for (auto it=begin(); it!=end(); ++it) {
        CNodeParams *params = (*it)->deepCopy();
        cgparams->paramslist.push_back(params);
    }
    return cgparams;
}

void CGraphParams::add(CNodeGrads &other, dtype_t scale) {
    auto it = this->begin();
    auto jt = other.begin();
    while (jt != other.end()) {
        if (it == this->end() || jt->first < (*it)->id) {
            assert(false);
        } else if (jt->first == (*it)->id) {
            mat_t &gradvec = *(jt->second);
            vec_t &parvec = (*it)->params;
            assert(gradvec.size2() == 1);
            assert(gradvec.size1() == parvec.size());
            for (int i=0; i<parvec.size(); ++i) {
                parvec(i) += scale*gradvec(i,0);
            }
            //soap::linalg::linalg_axpy(scale, gradvec, 
            //    parvec, parvec.size(), 1, 1, 0, 0);
            ++it;
            ++jt;
        } else if (jt->first > (*it)->id) {
            ++it;
        }
    }
}

void CGraphParams::addSquare(
        CNodeGrads &other, dtype_t coeff1, dtype_t coeff2) {
    auto it = this->begin();
    auto jt = other.begin();
    while (jt != other.end()) {
        if (it == this->end() || jt->first < (*it)->id) {
            assert(false);
        } else if (jt->first == (*it)->id) {
            mat_t &gradvec = *(jt->second);
            vec_t &parvec = (*it)->params;
            assert(gradvec.size2() == 1);
            assert(gradvec.size1() == parvec.size());
            for (int i=0; i<parvec.size(); ++i) {
                parvec(i) = coeff1*parvec(i) + coeff2*std::pow(gradvec(i,0), 2);
            }
            ++it;
            ++jt;
        } else if (jt->first > (*it)->id) {
            ++it;
        }
    }
}

void CGraphParams::add(
        CNodeGrads &other, dtype_t coeff, CGraphParams &frictions) {
    auto it = this->begin();
    auto ft = frictions.begin();
    auto jt = other.begin();
    while (jt != other.end()) {
        if (it == this->end() || jt->first < (*it)->id) {
            assert(false);
        } else if (jt->first == (*it)->id) {
            mat_t &gradvec = *(jt->second);
            vec_t &parvec = (*it)->params;
            vec_t &frictvec = (*ft)->params;
            assert(gradvec.size2() == 1);
            assert(gradvec.size1() == parvec.size());
            for (int i=0; i<parvec.size(); ++i) {
                parvec(i) += coeff*gradvec(i,0)/(std::sqrt(frictvec(i))+1e-10); // TODO epsilon
            }
            ++it;
            ++ft;
            ++jt;
        } else if (jt->first > (*it)->id) {
            ++it;
            ++ft;
        }
    }
}

// =====
// CNODE
// =====

CNode::CNode()
    : tag(""), id(-1), op("?"), active(true), 
      params(NULL), params_constant(false) {
    ;
}

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
}

void CNode::zero() {
    for (int i=0; i<vals.size(); ++i) vals(i) = 0.;
    grads.clear();
}

void CNode::resetAndResize() {
    if (inputs.size() > 0) {
        int n_slots = vals.size();
        int n_slots_req = inputs[0]->vals.size();
        if (n_slots != n_slots_req)
            this->resize(n_slots_req);
    }
    this->zero();
}

void CNode::feed(mat_t &X, int colidx, bool with_input_grads) {
    if (vals.size() != X.size1()) this->resize(X.size1());
    for (int i=0; i<X.size1(); ++i) {
        vals(i) = X(i,colidx);
    }
    if (with_input_grads) {
        input_grads.clear();
        mat_t &igrads = *(input_grads.allocate(id, X.size1(), 1));
        for (int i=0; i<X.size1(); ++i) {
            igrads(0,i) = 1.;
        }
    }
}

bpy::object CNode::valsNumpy(std::string np_dtype) {
    soap::linalg::numpy_converter npc(np_dtype.c_str());
    return npc.ublas_to_numpy<dtype_t>(vals);
}

void CNode::evaluate(bool with_input_grads) {
    assert(false);
}

void CNode::linkPython(bpy::list &py_nodelist) {
    CGraph::nodelist_t nodelist;
    for (int i=0; i<boost::python::len(py_nodelist); ++i) {
        nodelist.push_back(boost::python::extract<CNode*>(py_nodelist[i]));
    }
    this->link(nodelist);
}

void CNode::link(CGraph::nodelist_t &nodelist) {
    if (params != NULL)
        assert(nodelist.size() == inputs.size() 
            && "Relink only permitted with same number of inputs.");
    inputs.clear();
    for (auto it=nodelist.begin(); it!=nodelist.end(); ++it) 
        inputs.push_back(*it);
}

void CNode::setBranchActive(bool set_active) {
    this->setActive(set_active);
    for (auto par: inputs) par->setBranchActive(set_active);
}

bpy::list CNode::getInputsPython() {
    bpy::list list;
    for (auto par: inputs) list.append(par);
    return list;
}

void CNode::registerPython() {
    using namespace boost::python;
    class_<CNode, CNode*>("CNode", init<>())
        .def("vals", &CNode::valsNumpy)
        .def("grads", &CNode::getGrads, 
            return_value_policy<reference_existing_object>())
        .def("input_grads", &CNode::getInputGrads, 
            return_value_policy<reference_existing_object>())
        .def("params", &CNode::getParams, 
            return_value_policy<reference_existing_object>())
        .def("setParamsConstant", &CNode::setParamsConstant)
        .def("isParamsConstant", &CNode::isParamsConstant)
        .def("inputs", &CNode::getInputsPython)
        .def("link", &CNode::linkPython)
        .def("setBranchActive", &CNode::setBranchActive)
        .def("evaluate", &CNode::evaluate)
        .add_property("tag", &CNode::getTag, &CNode::setTag)
        .add_property("active", &CNode::isActive, &CNode::setActive)
        .add_property("size", &CNode::inputSize)
        .add_property("op", &CNode::getOp);
    class_<CNodeGrads, CNodeGrads*>("CNodeGrads", init<>())
        .add_property("size", &CNodeGrads::size)
        .def("vals", &CNodeGrads::valsNumpy)
        .def("listParamSets", &CNodeGrads::listParamSets);
    class_<Optimizer, Optimizer*>("CGraphOptimizer", init<Options*>()) 
        .def(init<std::string>())
        .def(init<>())
        .def("step", &Optimizer::stepNumpy)
        .def("fit", &Optimizer::fitNumpy);
}

void CNodeInput::evaluate(bool with_input_grads) {
    if (active) ;
    else {
        this->zero();
    }
}

void CNodeSigmoid::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    // Evaluate output
    vec_t &p = params->params;
    int n_slots = vals.size();
    int n_params = p.size();
    int n_inputs = inputs.size();
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
    if (!params->isConstant()) {
        mat_t &pgrads = *(grads.allocate(params->id, n_slots, n_params));
        for (int i=0; i<n_inputs; ++i) {
            vec_t &vi = inputs[i]->vals;
            for (int a=0; a<n_slots; ++a) {
                pgrads(i,a) = slot_scale(a)*vi(a);
            }
        }
        for (int a=0; a<n_slots; ++a) {
            pgrads(n_inputs,a) = slot_scale(a)*1.; // Last parameter is bias
        }
    }
    for (int i=0; i<n_inputs; ++i) {
        vec_t grad_slot_scale = p(i)*slot_scale;
        grads.add(inputs[i]->grads, grad_slot_scale);
    }
    if (with_input_grads) {
        input_grads.clear();
        for (int i=0; i<n_inputs; ++i) {
            vec_t grad_slot_scale = p(i)*slot_scale;
            input_grads.add(inputs[i]->input_grads, grad_slot_scale);
        }
    }
}

void CNodeExp::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    vec_t &p = params->params;
    vec_t &x = inputs[0]->vals;
    for (int a=0; a<vals.size(); ++a) {
        vals(a) = std::exp(p(0)*x(a));
    }
    if (!params->isConstant()) {
        mat_t &pgrads = *(grads.allocate(params->id, vals.size(), p.size()));
        for (int a=0; a<vals.size(); ++a) {
            pgrads(0,a) = x(a)*vals(a);
        }
    }
    vec_t grad_slot_scale = p(0)*vals;
    grads.add(inputs[0]->grads, grad_slot_scale);
    if (with_input_grads) {
        input_grads.clear();
        input_grads.add(inputs[0]->input_grads, grad_slot_scale);
    }
}

void CNodeLog::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    vec_t &x = inputs[0]->vals;
    vec_t grad_slot_scale = vec_t(vals.size(), 0.);
    for (int a=0; a<vals.size(); ++a) {
        vals(a) = std::log(x(a));
        grad_slot_scale(a) = 1./x(a);
    }
    grads.add(inputs[0]->grads, grad_slot_scale);
    if (with_input_grads) {
        input_grads.clear();
        input_grads.add(inputs[0]->input_grads, grad_slot_scale);
    }
}

void CNodeMod::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    vec_t &x = inputs[0]->vals;
    vec_t grad_slot_scale = vec_t(vals.size(), 0.);
    for (int a=0; a<vals.size(); ++a) {
        vals(a) = std::abs(x(a));
        grad_slot_scale(a) = (x(a) >= 0.) ? 1. : -1.;
    }
    grads.add(inputs[0]->grads, grad_slot_scale);
    if (with_input_grads) {
        input_grads.clear();
        input_grads.add(inputs[0]->input_grads, grad_slot_scale);
    }
}

void CNodePow::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    vec_t &p = params->params;
    vec_t &x = inputs[0]->vals;
    vec_t grad_slot_scale = vec_t(vals.size(), 0.);
    for (int a=0; a<vals.size(); ++a) {
        vals(a) = std::pow(x(a), p(0));
        if (std::abs(x(a)) < 1e-20) grad_slot_scale(a) = 0.; // TODO Define eps=1e-20
        else grad_slot_scale(a) = p(0)*std::pow(x(a), p(0)-1);
    }
    if (!params->isConstant()) {
        mat_t &pgrads = *(grads.allocate(params->id, vals.size(), p.size()));
        for (int a=0; a<vals.size(); ++a) {
            if (x(a) < 1e-20) pgrads(0,a) = 0.; // TODO Define eps=1e-20
            else pgrads(0,a) = vals(a)*std::log(x(a));
        }
    }
    grads.add(inputs[0]->grads, grad_slot_scale);
    if (with_input_grads) {
        input_grads.clear();
        input_grads.add(inputs[0]->input_grads, grad_slot_scale);
    }
}

void CNodeMult::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    vec_t &u = inputs[0]->vals;
    vec_t &v = inputs[1]->vals;
    for (int a=0; a<vals.size(); ++a) {
        vals(a) = u(a)*v(a);
    }
    grads.add(inputs[0]->grads, v);
    grads.add(inputs[1]->grads, u);
    if (with_input_grads) {
        input_grads.clear();
        input_grads.add(inputs[0]->input_grads, v);
        input_grads.add(inputs[1]->input_grads, u);
    }
}

void CNodeDiv::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    vec_t &u = inputs[0]->vals;
    vec_t &v = inputs[1]->vals;
    vec_t su = vec_t(vals.size(), 0.);
    vec_t sv = vec_t(vals.size(), 0.);
    for (int a=0; a<vals.size(); ++a) {
        vals(a) = u(a)/v(a);
        su(a) = 1./v(a);
        sv(a) = -su(a)*vals(a);
    }
    grads.add(inputs[0]->grads, su);
    grads.add(inputs[1]->grads, sv);
    if (with_input_grads) {
        input_grads.clear();
        input_grads.add(inputs[0]->input_grads, su);
        input_grads.add(inputs[1]->input_grads, sv);
    }
}

void CNodeLinear::evaluate(bool with_input_grads) {
    this->resetAndResize();
    if (!active) return;
    // Evaluate output
    vec_t &p = params->params;
    int n_slots = vals.size();
    int n_params = p.size();
    int n_inputs = inputs.size();
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
    if (!params->isConstant()) {
        mat_t &pgrads = *(grads.allocate(params->id, vals.size(), p.size()));
        for (int i=0; i<inputs.size(); ++i) {
            vec_t &vi = inputs[i]->vals;
            for (int a=0; a<vals.size(); ++a) {
                pgrads(i,a) = vi(a);
            }
        }
        for (int a=0; a<vals.size(); ++a) {
            pgrads(inputs.size(),a) = 1.; // Last parameter is bias
        }
    }
    for (int i=0; i<n_inputs; ++i) {
        dtype_t grad_scale = p(i);
        grads.add(inputs[i]->grads, grad_scale);
    }
    if (with_input_grads) {
        input_grads.clear();
        for (int i=0; i<n_inputs; ++i) {
            dtype_t grad_scale = p(i);
            input_grads.add(inputs[i]->input_grads, grad_scale);
        }
    }
}

void CNodeXENT::evaluate(bool with_input_grads) {
    int n_inputs = inputs.size();
    assert(n_inputs == 2 && "XENT node only allows two inputs = [output,target]");
    int n_slots = inputs[0]->vals.size();
    assert(inputs[1]->vals.size() == n_slots && "XENT: Input vector size mismatch");
    // Allocate output
    this->resize(1);
    this->zero();
    if (!active) return;
    // Evaluate output
    vec_t grad_slot_scale(n_slots, 0.);
    vec_t &yp = inputs[0]->vals;
    vec_t &yt = inputs[1]->vals;
    for (int a=0; a<n_slots; ++a) {
        vals(0) += yt(a)*std::log(yp(a)) + (1.-yt(a))*std::log(1-yp(a));
        grad_slot_scale(a) = yt(a)/yp(a) - (1.-yt(a))/(1.-yp(a));
    }
    vals *= -1./n_slots;
    grad_slot_scale *= -1./n_slots;
    // Evaluate gradients
    grads.add(inputs[0]->grads, grad_slot_scale);
    grads.sumSlots();
}

void CNodeMSE::evaluate(bool with_input_grads) {
    int n_inputs = inputs.size();
    assert(n_inputs == 2 && "MSE node only allows two inputs = [output,target]");
    int n_slots = inputs[0]->vals.size();
    assert(inputs[1]->vals.size() == n_slots && "MSE: Input vector size mismatch");
    // Allocate output
    this->resize(1);
    this->zero();
    // Check activity
    if (!active) {
        return;
    }
    // Evaluate output
    vec_t grad_slot_scale(n_slots, 0.);
    vec_t &y0 = inputs[0]->vals;
    vec_t &y1 = inputs[1]->vals;
    for (int a=0; a<n_slots; ++a) {
        vals(0) += std::pow(y0(a)-y1(a),2);
        grad_slot_scale(a) = y0(a)-y1(a);
    }
    vals *= 1./n_slots;
    grad_slot_scale *= 2./n_slots;
    // Evaluate gradients
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
        mat_t *reduced = new mat_t(full->size1(), 1, 0.);
        for (int i=0; i<full->size1(); ++i) {
            for (int j=0; j<full->size2(); ++j) {
                (*reduced)(i,0) += (*full)(i,j);
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
            int n_rows = jt->second->size1();
            int n_cols = jt->second->size2();
            // TODO Accelerate >>>
            // Manual code:
            // >>> mat_t *add_mat = new mat_t(n_rows, n_cols, 0);
            // >>> for (int i=0; i<jt->second->size2(); ++i) {
            // >>>     for (int j=0; j<jt->second->size1(); ++j) {
            // >>>         (*add_mat)(j,i) = slot_scale(i)*(*(jt->second))(j,i);
            // >>>     }
            // >>> }
            mat_t *add_mat;
            if (n_rows > n_cols) {
                add_mat = new mat_t(n_rows, n_cols, 0);
                for (int j=0; j<n_cols; ++j) {
                    soap::linalg::linalg_axpy(slot_scale[j], *(jt->second), 
                        *add_mat, n_rows, n_cols, n_cols, j, j);
                }
            } else {
                add_mat = new mat_t(n_rows, n_cols);
                for (int i=0; i<n_rows; ++i) {
                    soap::linalg::linalg_mul(*(jt->second), slot_scale, *add_mat, n_cols, i*n_cols, 0, i*n_cols);
                }
            }
            // <<< Accelerate TODO
            add_entries.push_back(paramset_t(jt->first, add_mat));
            ++jt;
        } else if (jt->first == it->first) {
            // TODO Accelerate >>>
            // Manual code:
            // >>> for (int i=0; i<jt->second->size2(); ++i) {
            // >>>     for (int j=0; j<jt->second->size1(); ++j) {
            // >>>         (*(it->second))(j,i) += slot_scale(i)*(*(jt->second))(j,i);
            // >>>     }
            // >>> }
            int n_rows = jt->second->size1();
            int n_cols = jt->second->size2();
            if (n_rows > n_cols) {
                for (int j=0; j<n_cols; ++j) {
                    soap::linalg::linalg_axpy(slot_scale[j], *(jt->second), 
                        *(it->second), n_rows, n_cols, n_cols, j, j);
                }
            } else {
                vec_t tmp(n_cols);
                for (int i=0; i<n_rows; ++i) {
                    soap::linalg::linalg_mul(*(jt->second), slot_scale, tmp, 
                        n_cols, i*n_cols, 0, 0);
                    soap::linalg::linalg_axpy(1., tmp, *(it->second),
                        n_cols, 1, 1, 0, i*n_cols);
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
            // TODO Accelerate >>>
            // Manual code:
            // >>> mat_t *add_mat = new mat_t(jt->second->size1(), jt->second->size2());
            // >>> *add_mat = scale*(*(jt->second));
            mat_t *add_mat = new mat_t(jt->second->size1(), jt->second->size2(), 0.);
            soap::linalg::linalg_axpy(scale, *(jt->second), *add_mat, 
                add_mat->size1()*add_mat->size2(), 1, 1, 0, 0);
            // <<< Accelerate TODO
            add_entries.push_back(paramset_t(jt->first, add_mat));
            ++jt;
        } else if (jt->first == it->first) {
            // TODO Accelerate >>>
            // Manual code:
            // >>> (*(it->second)) += scale*(*(jt->second));
            soap::linalg::linalg_axpy(scale, *(jt->second), *(it->second), 
                it->second->size1()*it->second->size2(), 1, 1, 0, 0);
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

mat_t *CNodeGrads::allocate(int params_id, int n_slots, int n_params) {
    assert(sparse_id_map.find(params_id) == sparse_id_map.end());
    mat_t *new_mat = new mat_t(n_params, n_slots, 0.);   
    sparse_grads.push_back(paramset_t(params_id, new_mat));
    sparse_id_map[params_id] = true;
    this->sort();
    return new_mat;
}

void CNodeGrads::listParamSets() {
    GLOG() << "Dependent on " << sparse_grads.size() << " parameter sets" << std::endl;
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

Optimizer::Optimizer() : options(NULL) {
    optalg = OptimizerCreator().create("adagrad");
}

Optimizer::Optimizer(std::string method) : options(NULL) {
    optalg = OptimizerCreator().create(method);
}

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

void Optimizer::stepNumpy(CGraph *cgraph, bpy::object &np_X, bpy::object &np_Y,
        std::string np_dtype_X, std::string np_dtype_Y,
        int n_steps, double rate) {
    mat_t X;
    soap::linalg::numpy_converter npc_X(np_dtype_X.c_str());
    npc_X.numpy_to_ublas<dtype_t>(np_X, X);
    mat_t Y;
    soap::linalg::numpy_converter npc_Y(np_dtype_Y.c_str());
    npc_Y.numpy_to_ublas<dtype_t>(np_Y, Y);
    optalg->step(cgraph, X, Y, n_steps, rate);
}

OptAdaGrad::~OptAdaGrad() {
    if (frictions) delete frictions;
    frictions = NULL;
}

void OptAdaGrad::fit(CGraph *cgraph, mat_t &X, mat_t &Y) {
    assert(false && "Use python wrapper in soap.soapy.cgraph");
}

void OptAdaGrad::step(CGraph *cgraph, mat_t &X, mat_t &Y, int n_steps, double rate) {
    if (frictions == NULL) {
        frictions = cgraph->getParams().deepCopy();
    }
    bool silent = GLOG.isSilent();
    if (!silent) GLOG.toggleSilence();
    for (int n=0; n<n_steps; ++n) {
        cgraph->feed(X, Y, false);
        CGraphParams &params = cgraph->getParams();
        for (auto it=cgraph->beginObjectives(); it!=cgraph->endObjectives(); ++it) {
            CNode *eps = *it;
            GLOG() << "Objective = " << eps->vals(0) << std::endl;
            CNodeGrads &grads = eps->getGrads();
            frictions->addSquare(grads, 1., 1.);
            params.add(grads, -1.*rate, *frictions);
        }
    }
    if (!silent) GLOG.toggleSilence();
    for (auto it=cgraph->beginObjectives(); it!=cgraph->endObjectives(); ++it) {
        CNode *eps = *it;
        GLOG() << op << "   +" << n_steps << " steps : " << eps->getOp() 
            << " = " << (boost::format("%1$1.4e") % eps->vals(0)).str() << std::endl;
    }
}

void OptSteep::fit(CGraph *cgraph, mat_t &X, mat_t &Y) {
    assert(false && "Use python wrapper in soap.soapy.cgraph");
}

void OptSteep::step(CGraph *cgraph, mat_t &X, mat_t &Y, int n_steps, double rate) {
    bool silent = GLOG.isSilent();
    if (!silent) GLOG.toggleSilence();
    for (int n=0; n<n_steps; ++n) {
        cgraph->feed(X, Y, false);
        CGraphParams &params = cgraph->getParams();
        for (auto it=cgraph->beginObjectives(); it!=cgraph->endObjectives(); ++it) {
            CNode *eps = *it;
            GLOG() << "Objective = " << eps->vals(0) << std::endl;
            CNodeGrads &grads = eps->getGrads();
            params.add(grads, -1.*rate);
        }
    }
    if (!silent) GLOG.toggleSilence();
    for (auto it=cgraph->beginObjectives(); it!=cgraph->endObjectives(); ++it) {
        CNode *eps = *it;
        GLOG() << "+ " << n_steps << " steps : " << eps->getOp() << " = " << eps->vals(0) << std::endl;
    }
}

void CNodeFactory::registerAll(void) {
	CNodeCreator().Register<CNodeInput>("input");
	CNodeCreator().Register<CNodeExp>("exp");
	CNodeCreator().Register<CNodeLog>("log");
	CNodeCreator().Register<CNodeMod>("mod");
	CNodeCreator().Register<CNodePow>("pow");
	CNodeCreator().Register<CNodeMult>("mult");
	CNodeCreator().Register<CNodeDiv>("div");
	CNodeCreator().Register<CNodeLinear>("linear");
	CNodeCreator().Register<CNodeSigmoid>("sigmoid");
	CNodeCreator().Register<CNodePearson>("pearson");
	CNodeCreator().Register<CNodeMSE>("mse");
	CNodeCreator().Register<CNodeXENT>("xent");
}

void OptimizerFactory::registerAll(void) {
    OptimizerCreator().Register<OptAdaGrad>("adagrad");
    OptimizerCreator().Register<OptSteep>("steep");
}

}}

