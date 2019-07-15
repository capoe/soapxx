import numpy as np
import soap
import momo
log = momo.osio
np_dtype = "float64"

def require(props, key, default=None, log=log):
    if default is None and not key in props:
        log << log.mr << "ERROR Missing property '%s'" % key << log.endl
        raise ValueError("Abort due to missing property field.")
    elif not key in props:
        return default
    return props[key]

class PyNodeParams(object):
    def __init__(self, shape):
        self.C = np.zeros(shape, np_dtype)
        self.grad = None
        self.friction = None
        self.shape = self.C.shape
    def randomize(self):
        self.C = np.random.uniform(-1., 1., size=self.shape) 
    def set(self, C):
        self.C = np.copy(C)
    def zeroGrad(self):
        self.grad = np.zeros(self.C.shape, np_dtype)
    def addGrad(self, g):
        self.grad = self.grad + g
    def zeroFrictions(self):
        self.friction = np.zeros(self.C.shape, np_dtype)
    def incrementFrictions(self):
        if self.friction is None: self.zeroFrictions()
        self.friction = self.friction + self.grad**2
    def nParams(self):
        return self.C.size

class PyNode(object):
    def __init__(self, idx, parents, props):
        self.idx = idx
        self.dim = require(props, "dim", 1)
        self.tag = require(props, "tag", "")
        self.op = None
        self.parents = parents
        self.props = props
        self.deps = {}
        self.params = None
        self.X_in = np.array([])
        self.X_out = None
        self.active = True
        self.grad = None
    def setParams(self, p):
        self.params = p
    def setVals(self, X):
        self.X_out = X
    def val(self):
        return self.X_out
    def evaluate(self):
        return
    def calcParamsShape(self):
        return [0,0]
    def setDependencies(self):
        self.deps = { p.idx: p for p in self.parents }
        for p in self.parents:
            self.deps.update(p.deps)
        return self.deps
    def zeroGrad(self):
        self.grad = np.zeros(self.X_out.shape, np_dtype)
        return
    def backpropagate(self, g_back, level=0, log=None):
        raise NotImplementedError("Missing function overload in ::backpropagate")
        return
    def printInfo(self, log=log):
        log << "Have node: %3d %-10s %4s <- depends on %s" % (
            self.idx, self.op, self.tag, str(sorted(self.deps.keys()))) << log.endl
    def printDim(self, log=log):
        log << "Node %d '%s': %s => %s" % (
            self.idx, self.op, self.X_in.shape, self.X_out.shape) << log.endl

class PyNodeInput(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "input"
        return
    def backpropagate(self, g_back, level=0, log=None):
        if log:
            log << "  "*level << log.flush
            self.printDim()
        self.grad = self.grad + g_back
        return

class PyNodeScalar(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "scalar"
    def calcParamsShape(self):
        dim = sum([ p.dim for p in self.parents ])
        return (dim,)
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.X_out = self.X_in*self.params.C
    def backpropagate(self, g_back, level=0, log=None):
        g_C = np.sum(g_back*self.X_in, axis=0)
        g_X = g_back*self.params.C
        self.params.addGrad(g_C)
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeExp(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "exp"
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.X_out = np.exp(self.X_in)
    def backpropagate(self, g_back, level=0, log=None):
        g_X = g_back*self.X_out
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeLinear(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "linear"
    def calcParamsShape(self):
        dim1 = sum([ p.dim for p in self.parents ]) + 1
        dim2 = self.dim
        return [dim1,dim2]
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.X_out = self.X_in.dot(self.params.C[0:-1,:])+self.params.C[-1,:]
    def backpropagate(self, g_back, level=0, log=None):
        x0 = np.concatenate([self.X_in, np.ones((self.X_in.shape[0],1))], axis=1)
        g_C = x0.T.dot(g_back) # (dim_in+1) x dim_out
        self.params.addGrad(g_C)
        g_X = g_back.dot(self.params.C[0:-1,:].T) # n x dim_in
        if log:
            log << "  "*level << log.flush
            self.printDim(log)
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeSoftmax(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.dim = sum([ p.dim for p in self.parents ])
        self.op = "softmax"
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.X_out = np.exp(self.X_in)
        self.X_out = (self.X_out.T/np.sum(self.X_out, axis=1)).T
    def backpropagate(self, g_back, level=0, log=None):
        g0 = g_back*self.X_out
        g1 = - (self.X_out.T*np.sum(g_back*self.X_out, axis=1)).T
        g_X = g0+g1
        if log:
            log << "  "*level << log.flush
            self.printDim(log)
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeSigmoid(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "sigmoid"
    def calcParamsShape(self):
        dim1 = sum([ p.dim for p in self.parents ]) + 1
        dim2 = self.dim
        return [dim1,dim2]
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.X_out = 1./(1.+np.exp(self.X_in.dot(self.params.C[0:-1,:])+self.params.C[-1,:]))
    def backpropagate(self, g_back, level=0, log=None):
        x0 = np.concatenate([self.X_in, np.ones((self.X_in.shape[0],1))], axis=1)
        g = g_back*self.X_out*(self.X_out-1.) # N x dim_out
        g_C = x0.T.dot(g) # (dim_in+1) x dim_out
        self.params.addGrad(g_C)
        g_X = g.dot(self.params.C[0:-1,:].T) # n x dim_in
        if log:
            log << "  "*level << log.flush
            self.printDim(log)
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeDot(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.dim = 1
        self.op = "dot"
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_out = np.sum(self.parents[0].X_out*self.parents[1].X_out, axis=1).reshape((-1,1))
    def backpropagate(self, g_back, level=0, log=None):
        g0 = g_back*self.parents[1].X_out
        g1 = g_back*self.parents[0].X_out
        self.parents[0].backpropagate(g0, level=level+1, log=log)
        self.parents[1].backpropagate(g1, level=level+1, log=log)

class PyNodeMSE(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "mse"
    def evaluate(self):
        assert len(self.parents) == 2
        self.X_out = np.sum((self.parents[0].X_out - self.parents[1].X_out)**2) \
            /self.parents[0].X_out.shape[0]
    def backpropagate(self, g_back=1., level=0, log=None):
        if log:
            log << "  "*level << log.flush
            self.printDim()
        g_X = 2./self.parents[0].X_out.shape[0]*(self.parents[0].X_out - self.parents[1].X_out)
        self.parents[0].backpropagate(+1.*g_X*g_back, level=level+1, log=log)
        self.parents[1].backpropagate(-1.*g_X*g_back, level=level+1, log=log)

class PyNodeXENT(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "xent"
    def evaluate(self):
        assert len(self.parents) == 2
        yp = self.parents[0].X_out
        yt = self.parents[1].X_out
        self.X_out = -1./yt.shape[0]*np.sum(yt*np.log(yp) + (1.-yt)*np.log(1.-yp))
    def backpropagate(self, g_back=1., level=0, log=None):
        yp = self.parents[0].X_out
        yt = self.parents[1].X_out
        gp = -1./yt.shape[0]*(yt/yp - (1.-yt)/(1.-yp))
        gt = -1./yt.shape[0]*(np.log(yp) - np.log(1.-yp))
        self.parents[0].backpropagate(gp*g_back, level=level+1, log=log)
        self.parents[1].backpropagate(gt*g_back, level=level+1, log=log)

PyNode.prototypes = {
    "input": PyNodeInput,
    "scalar": PyNodeScalar,
    "exp": PyNodeExp,
    "linear": PyNodeLinear,
    "sigmoid": PyNodeSigmoid,
    "softmax": PyNodeSoftmax,
    "dot": PyNodeDot,
    "mse": PyNodeMSE,
    "xent": PyNodeXENT
}

class PyGraph(object):
    def __init__(self):
        self.nodes = []
        self.node_map = {}
        self.dependency_map = {}
        self.params = []
        self.params_map = {}
    def addNode(self, op, parents=[], props={}):
        idx = len(self.nodes)
        # Create node
        new_node = PyNode.prototypes[op](idx, parents, props)
        self.nodes.append(new_node)
        if "tag" in props:
            self.node_map[props["tag"]] = new_node
        # Trace dependencies
        new_node.setDependencies()
        # Allocate params
        params_tag = props["params"] if "params" in props else \
            "_params_%d" % new_node.idx
        params_shape = new_node.calcParamsShape()
        if params_tag in self.params_map:
            new_params = self.params_map[params_tag]
        else:
            new_params = PyNodeParams(params_shape)
            self.params.append(new_params)
        assert len(new_params.shape) == len(params_shape)
        for i in range(len(new_params.shape)):
            assert new_params.shape[i] == params_shape[i]
        new_node.params = new_params
        self.params_map[params_tag] = new_params
        return new_node
    def __getitem__(self, node_tag):
        return self.node_map[node_tag]
    def printInfo(self):
        n_params = sum([ p.nParams() for p in self.params ])
        log << "Graph with %d nodes and %d parameter sets with %d parameters" % (
            len(self.nodes), len(self.params_map), n_params) << log.endl
        for node in self.nodes:
            node.printInfo()
    def evaluate(self, node, feed):
        for node_tag, X in feed.iteritems():
            self.node_map[node_tag].setVals(X)
        path = [ n for nidx, n in node.deps.iteritems() ]
        path = sorted(path, key=lambda d: d.idx)
        for pathnode in path:
            pathnode.evaluate()
        node.evaluate()
        return node.X_out
    def propagate(self, feed, log=None):
        for node_tag, X in feed.iteritems():
            self.node_map[node_tag].setVals(X)
        for node in self.nodes:
            if log: log << log.back << "Propagate node %d/%d" % (
                node.idx+1, len(self.nodes)) << log.flush
            node.evaluate()
        if log: log << log.endl
    def backpropagate(self, node=None, log=None):
        for param in self.params: param.zeroGrad()
        for node in self.nodes: node.zeroGrad()
        if node is None: node = self.nodes[-1]
        node.backpropagate(log=log)

class PyNodeDropout(object):
    def __init__(self):
        return

class PyGraphOptimizer(object):
    def __init__(self):
        return
    def step(self, graph, feed, n_steps):
        return
    def optimize(self, graph):
        return

class OptAdaGrad(PyGraphOptimizer):
    def __init__(self, props):
        PyGraphOptimizer.__init__(self)
        self.type = "adagrad"
        self.rate = require(props, "rate")
        self.eps = require(props, "eps", 1e-10)
    def initialize(self, graph):
        for p in graph.params: p.zeroFrictions()
    def step(self, graph, feed, n_steps, obj, log=None, report_every=10):
        for n in range(n_steps):
            graph.propagate(feed=feed)
            graph.backpropagate(log=None)
            for par in graph.params:
                self.stepNodeParams(par)
            if log and (n % report_every == 0 or n == n_steps-1):
                log << "Step %3d: %s%s = %+1.4e" % (
                    n, obj.op, obj.tag, obj.val()) << log.endl
        return
    def stepNodeParams(self, params):
        params.incrementFrictions()
        params.C = params.C - 1.*self.rate*params.grad/(np.sqrt(params.friction)+self.eps)
    def optimize(self, graph, feed, n_steps):
        return

