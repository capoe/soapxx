import numpy as np
import soap
import momo
import pickle
import copy
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
    def __init__(self, tag, shape):
        self.tag = tag
        self.C = np.zeros(shape, np_dtype)
        self.grad = None
        self.friction = None
        self.shape = self.C.shape
        self.constant = False
    def randomize(self):
        self.C = np.random.uniform(-1., 1., size=self.shape).astype(np_dtype)
    def set(self, C):
        self.C = np.copy(C)
    def zeroGrad(self):
        if not self.constant:
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
    def purge(self, purge_frictions=True):
        self.grad = None
        if purge_frictions: self.friction = None

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
        self.friction = None
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
    def zeroFrictions(self):
        self.friction = np.zeros(self.X_out.shape, np_dtype)
    def incrementFrictions(self):
        if self.friction is None: self.zeroFrictions()
        self.friction = self.friction + self.grad**2
    def backpropagate(self, g_back, level=0, log=None):
        raise NotImplementedError("Missing function overload in ::backpropagate")
        return
    def printInfo(self, log=log):
        dep_str = str(sorted(self.deps.keys()))
        if len(dep_str) > 17: dep_str = "(%d nodes)" % (len(self.deps.keys()))
        log << "Have node: %3d %-10s %-20s  dim=%-4d <- depends on %s" % (
            self.idx, self.op, self.tag, self.dim, dep_str) << log.endl
    def printDim(self, log=log):
        log << "Node %d '%s': %s => %s" % (
            self.idx, self.op, self.X_in.shape, self.X_out.shape) << log.endl
    def purge(self):
        self.X_in = np.array([])
        self.X_out = None
        self.grad = None

class PyNodeInput(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "input"
    def backpropagate(self, g_back=1., level=0, log=None):
        self.grad = self.grad + g_back

class PyNodeFlexInput(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "flexinput"
    def zeroGrad(self):
        self.grad = []
        for row in range(self.X_out.shape[0]):
            self.grad.append(np.zeros_like(self.X_out[row]))
        self.grad = np.array(self.grad)
    def zeroFrictions(self):
        self.friction = []
        for row in range(self.X_out.shape[0]):
            self.friction.append(np.zeros_like(self.X_out[row]))
        self.friction = np.array(self.friction)
    def incrementFrictions(self):
        if self.friction is None: self.zeroFrictions()
        for row in range(self.X_out.shape[0]):
            self.friction[row] = self.friction[row] + self.grad[row]**2
    def backpropagate(self, g_back=1., level=0, log=None):
        for row in range(len(g_back)):
            self.grad[row] = self.grad[row] + g_back[row]

class PyNodeJoin(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.dim = sum([ p.dim for p in parents ])
        self.op = "join"
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_out = np.concatenate([ p.X_out for p in self.parents ], axis=1)
    def backpropagate(self, g_back=1., level=0, log=None):
        off = 0
        for p in self.parents:
            p.backpropagate(g_back[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeSlice(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.slice = require(props, "slice")
        self.dim = self.slice[1]-self.slice[0]
        self.op = "slice"
        assert len(parents) == 1
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_out = self.parents[0].X_out[:,self.slice[0]:self.slice[1]]
    def backpropagate(self, g_back=1., level=0, log=None):
        g = np.zeros((self.parents[0].X_out.shape[1],))
        g[self.slice[0]:self.slice[1]] = g_back
        self.parents[0].backpropagate(g, level=level+1, log=log)
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
        g_X = g_back*self.params.C
        if not self.params.constant:
            g_C = np.sum(g_back*self.X_in, axis=0)
            self.params.addGrad(g_C)
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeAdd(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "add"
        self.dim = parents[0].dim
        for p in parents: assert p.dim == parents[0].dim
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_out = np.sum([ p.X_out for p in self.parents ], axis=0)
    def backpropagate(self, g_back, level=0, log=None):
        for p in self.parents: p.backpropagate(g_back, level=level+1, log=log)

class PyNodeMult(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "mult"
        self.dim = parents[0].dim
        assert len(parents) == 2
        assert parents[0].dim == parents[1].dim
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_out = self.parents[0].X_out*self.parents[1].X_out
    def backpropagate(self, g_back, level=0, log=None):
        g0 = g_back*self.parents[1].X_out
        g1 = g_back*self.parents[0].X_out
        self.parents[0].backpropagate(g0, level=level+1, log=log)
        self.parents[1].backpropagate(g1, level=level+1, log=log)

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
        if not self.params.constant:
            g_C = x0.T.dot(g_back) # (dim_in+1) x dim_out
            self.params.addGrad(g_C)
        g_X = g_back.dot(self.params.C[0:-1,:].T) # n x dim_in
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
        if not self.params.constant:
            g_C = x0.T.dot(g) # (dim_in+1) x dim_out
            self.params.addGrad(g_C)
        g_X = g.dot(self.params.C[0:-1,:].T) # n x dim_in
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

class PyNodeOuter(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)

class PyNodeMSE(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "mse"
        assert len(self.parents) == 2
    def evaluate(self):
        N = self.parents[0].X_out.shape[0]
        if len(self.parents[0].X_out.shape) > 1: N *= self.parents[0].X_out.shape[1]
        self.X_out = np.sum((self.parents[0].X_out - self.parents[1].X_out)**2)/N
    def backpropagate(self, g_back=1., level=0, log=None):
        N = self.parents[0].X_out.shape[0]
        if len(self.parents[0].X_out.shape) > 1: N *= self.parents[0].X_out.shape[1]
        g_X = 2./N*(self.parents[0].X_out - self.parents[1].X_out)
        self.parents[0].backpropagate(+1.*g_X*g_back, level=level+1, log=log)
        self.parents[1].backpropagate(-1.*g_X*g_back, level=level+1, log=log)

class PyNodeXENT(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "xent"
        assert len(self.parents) == 2
    def evaluate(self):
        yp = self.parents[0].X_out
        yt = self.parents[1].X_out
        N = yt.shape[0]
        if len(yt.shape) > 1: N *= yt.shape[1]
        self.X_out = -1./N*np.sum(yt*np.log(yp) + (1.-yt)*np.log(1.-yp))
    def backpropagate(self, g_back=1., level=0, log=None):
        yp = self.parents[0].X_out
        yt = self.parents[1].X_out
        N = yt.shape[0]
        if len(yt.shape) > 1: N *= yt.shape[1]
        gp = -1./N*(yt/yp - (1.-yt)/(1.-yp))
        gt = -1./N*(np.log(yp) - np.log(1.-yp))
        self.parents[0].backpropagate(gp*g_back, level=level+1, log=log)
        self.parents[1].backpropagate(gt*g_back, level=level+1, log=log)

class PyNodeStochasticStep(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "sstep"
        self.dim = sum([ p.dim for p in parents ])
        #self.sigma = props["sigma"]
        self.random_state = None
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.random_state = np.random.uniform(size=self.X_in.shape)
        self.X_out = np.heaviside(self.random_state - self.X_in, 0.0)
    def backpropagate(self, g_back, level=0, log=None):
        g_X = -g_back #+ 0.0000001*np.log(self.X_in/(1.-self.X_in))    #* np.exp(-0.5*(self.random_state-self.X_in)**2/self.sigma**2)
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeStochasticSoftStep(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "ssoftstep"
        self.dim = sum([ p.dim for p in parents ])
        self.alpha = 1./props["sigma"]
        self.rng = props["rng"]
        self.z0 = self.rng[0]
        self.dz = self.rng[1] - self.rng[0]
        self.phi = None
        self.random_state = None
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.random_state = np.random.uniform(size=self.X_in.shape)
        self.phi = 1./(1. + np.exp(-self.alpha*(self.random_state - self.X_in)))
        self.X_out = self.z0 + self.dz*self.phi
        #self.X_out = self.z0 + self.dz*np.heaviside(self.random_state - self.X_in, 0.0)
    def backpropagate(self, g_back, level=0, log=None):
        #g_X = -g_back*self.dz*self.phi*(1.-self.phi)*self.alpha
        g_X = -g_back*self.dz*self.phi*(1.-self.phi)*self.alpha #*np.exp(-0.5*self.alpha**2*(self.random_state-self.X_in)**2)
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

class PyNodeStochasticSphericalGaussian(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "ssgaussian"
        assert parents[0].dim == parents[1].dim
        self.dim = parents[0].dim
        self.random_state = None
        self.kl_weight = props["kl_weight"]
        assert len(self.parents) == 2 # mu, std
    def calcParamsShape(self):
        return [0,0]
    def evaluate(self):
        shape_out = self.parents[0].X_out.shape
        self.random_state = np.random.normal(loc=0.0, scale=1.0, size=shape_out)
        self.X_in = None
        self.X_out = self.parents[0].X_out + self.random_state*self.parents[1].X_out # mu + std*eps
    def backpropagate(self, g_back, level=0, log=None):
        mu = self.parents[0].X_out
        std = self.parents[1].X_out
        g_mu = g_back
        g_std = g_back*self.random_state
        g_mu = g_mu + self.kl_weight*mu
        g_std = g_std + self.kl_weight*(std - 1./(std+1e-10))
        self.parents[0].backpropagate(g_mu, level=level+1, log=log)
        self.parents[1].backpropagate(g_std, level=level+1, log=log)

class PyNodeConstant(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "constant"
        self.dim = props["dim"]
    def calcParamsShape(self):
        return (self.dim,)
    def evaluate(self):
        self.X_out = np.tile(self.params.C, (self.parents[0].X_out.shape[0],1))
    def backpropagate(self, g_back, level, log):
        self.params.addGrad(np.sum(g_back, axis=0))

class PyNodeReLu(PyNode):
    def __init__(self, idx, parents, props):
        PyNode.__init__(self, idx, parents, props)
        self.op = "relu"
        #self.dim = sum([ p.dim for p in parents ])
    #def evaluate(self):
    #    self.X_out = np.concatenate([ np.maximum(p.X_out, 0.0) for p in self.parents ], axis=1)
    #def backpropagate(self, g_back, level, log):
    #    for p in self.parents:
    #        p.backpropagate(g_back*np.heaviside(p.X_out, 0.0), level=level+1, log=log)
    def calcParamsShape(self):
        dim1 = sum([ p.dim for p in self.parents ]) + 1
        dim2 = self.dim
        return [dim1,dim2]
    def evaluate(self):
        self.X_in = np.concatenate([ node.X_out for node in self.parents ], axis=1)
        self.X_out = self.X_in.dot(self.params.C[0:-1,:])+self.params.C[-1,:]
        self.X_out = np.maximum(self.X_out, 0.0)
    def backpropagate(self, g_back, level=0, log=None):
        mask = np.heaviside(self.X_out, 0.0)
        x0 = np.concatenate([self.X_in, np.ones((self.X_in.shape[0],1))], axis=1)
        if not self.params.constant:
            g_C = x0.T.dot(mask*g_back) # (dim_in+1) x dim_out
            self.params.addGrad(g_C)
        g_X = (mask*g_back).dot(self.params.C[0:-1,:].T) # n x dim_in
        off = 0
        for p in self.parents:
            p.backpropagate(g_X[:,off:off+p.dim], level=level+1, log=log)
            off += p.dim

PyNode.prototypes = {
    "input":      PyNodeInput,
    "flexinput":  PyNodeFlexInput,
    "constant":   PyNodeConstant,
    "slice":      PyNodeSlice,
    "join":       PyNodeJoin,
    "scalar":     PyNodeScalar,
    "add":        PyNodeAdd,
    "mult":       PyNodeMult,
    "exp":        PyNodeExp,
    "linear":     PyNodeLinear,
    "relu":       PyNodeReLu,
    "sigmoid":    PyNodeSigmoid,
    "softmax":    PyNodeSoftmax,
    "dot":        PyNodeDot,
    "mse":        PyNodeMSE,
    "xent":       PyNodeXENT,
    "sstep":      PyNodeStochasticStep,
    "ssoftstep":  PyNodeStochasticSoftStep,
    "ssgaussian": PyNodeStochasticSphericalGaussian
}

class PyGraph(object):
    def __init__(self):
        self.nodes = []
        self.node_map = {}
        self.dependency_map = {}
        self.params = []
        self.params_map = {}
        self.meta = {}
        self.chks = []
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
            new_params = PyNodeParams(tag=params_tag, shape=params_shape)
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
        if node is None: node = self.nodes[-1]
        for node_idx, dep in node.deps.iteritems():
            dep.zeroGrad()
        node.backpropagate(log=log)
    def purge(self, purge_frictions=True):
        for node in self.nodes: node.purge()
        for par in self.params: par.purge(purge_frictions=purge_frictions)
    def checkpoint(self, chkfile=None, info={}, log=None):
        if log: log << log.mb << "Checkpoint #%d ..." % len(self.chks) << log.flush
        self.purge(purge_frictions=False)
        cpy = copy.deepcopy(self)
        cpy.chks = []
        cpy.meta.update(info)
        self.chks.append(cpy)
        if chkfile is not None: 
            if log: log << log.mb << "writing '%s' ..." % chkfile << log.flush
            cpy.save(chkfile)
        if log: log << log.mb << "done" << log.endl
    def save(self, archfile):
        with open(archfile, 'w') as f:
            f.write(pickle.dumps(self))
    def load(self, archfile):
        self = pickle.load(open(archfile, 'rb')) 
        return self 

class PyNodeDropout(object):
    def __init__(self):
        return

class Subsampler(object):
    def __init__(self, feed, idx_map={}):
        self.feed = feed
        self.idx_map = idx_map
        self.max = self.findMaxRange(feed, idx_map)
        self.lambdas = self.setupLambdas(feed, idx_map)
        self.type = "none"
    def sample(self, n_sub):
        raise NotImplementedError("Missing ::sample implementation for '%s'" % self.type)
    def setupLambdas(self, feed, idx_map):
        lambdas = {}
        for addr in feed:
            if addr in idx_map:
                lambdas[addr] = lambda addr, idcs, X: X[self.idx_map[addr][idcs]]
            else:
                lambdas[addr] = lambda addr, idcs, X: X[idcs]
        return lambdas
    def findMaxRange(self, feed, idx_map):
        test_key = self.feed.keys()[0]
        if test_key in idx_map: return idx_map[test_key].shape[0]
        else: return self.feed[test_key].shape[0]

class RandReSubsampler(Subsampler):
    def __init__(self, feed, idx_map={}):
        Subsampler.__init__(self, feed, idx_map)
        self.type = "randre"
    def sample(self, n_sub):
        select = np.random.randint(0, self.max, n_sub)
        subfeed = { addr: self.lambdas[addr](addr, select, x) \
            for addr, x in self.feed.iteritems() }
        return subfeed

class PyGraphOptimizer(object):
    def __init__(self):
        self.type = "none"
    def step(self, graph, feed, n_steps):
        raise NotImplementedError(
            "Missing ::step implementation for '%s'" % self.type)
    def stepNodeParams(self, params):
        raise NotImplementedError(
            "Missing ::stepNodeParams implementation for '%s'" % self.type)
    def initialize(self, graph):
        return
    def initializeFitInput(self, inputs):
        return
    def fit(self, graph, n_iters, n_batch, n_steps, feed=None,
            idx_map={}, report_every=-1,
            subsampler=None, log=None, verbose=False, chk_every=-1, chkfile=None):
        if chk_every > 0: assert chkfile != None
        if subsampler is None: 
            assert feed is not None
            subsampler = RandReSubsampler(feed, idx_map=idx_map)
            log << "Using random resampler over %d samples from data feed" % (
                subsampler.max) << log.endl
        report_on = graph["loss"] if "loss" in graph.node_map else graph.nodes[-1]
        for it in range(n_iters):
            subfeed = subsampler.sample(n_batch)
            if log: 
                log << "Batch %-d/%d:" % (it+1, n_iters) << log.flush
                for key in sorted(subfeed):
                    log << "(%s)=%s" % (key, subfeed[key].shape) << log.flush
                log << log.endl
            self.step(graph=graph, n_steps=n_steps, 
                report_every=report_every,
                obj=report_on,
                feed=subfeed,
                log=log if verbose else None)
            if log: log << " => %s%s = %+1.4e" % (
                report_on.op, report_on.tag, report_on.val()) << log.endl
            if chk_every > 0 and it > 0 and (it % chk_every == 0):
                graph.checkpoint(info={"iter":it}, chkfile=chkfile, log=log)
    def fitInput(self, graph, feed, n_iters, n_steps, inputs, postprocess=None, obj=None):
        if obj is None: obj = graph["loss"]
        graph.propagate(feed=feed)
        self.initializeFitInput(inputs)
        for it in range(n_iters):
            log << "Iteration %d" % it << log.endl
            for n in range(n_steps):
                graph.propagate(feed={})
                graph.backpropagate(log=None)
                for node in inputs:
                    self.stepInput(node)
                if postprocess is not None:
                    postprocess(graph)
                log << "  Step %3d: %s%s = %+1.4e" % (
                    n, obj.op, obj.tag, obj.val()) << log.endl
    def save(self, archfile):
        with open(archfile, 'w') as f:
            f.write(pickle.dumps(self))
    def load(self, archfile):
        self = pickle.load(open(archfile, 'rb')) 
        return self 

class OptAdaGrad(PyGraphOptimizer):
    def __init__(self, props={}):
        PyGraphOptimizer.__init__(self)
        self.type = "adagrad"
        self.rate = require(props, "rate", 0.01)
        self.eps = require(props, "eps", 1e-10)
        self.datalog = {"loss":[]}
    def initialize(self, graph):
        for p in graph.params: p.zeroFrictions()
    def initializeFitInput(self, inputs):
        for i in inputs: i.zeroFrictions()
    def step(self, graph, feed, n_steps, obj, log=None, report_every=10):
        for n in range(n_steps):
            graph.propagate(feed=feed)
            graph.backpropagate(log=None)
            for par in graph.params:
                self.stepNodeParams(par)
            self.datalog["loss"].append([n, obj.val()])
            if log and (n % report_every == 0 or n == n_steps-1):
                log << "  Step %3d: %s%s = %+1.4e" % (
                    n, obj.op, obj.tag, obj.val()) << log.endl
        return
    def stepNodeParams(self, params):
        if params.constant: raise RuntimeError()
        params.incrementFrictions()
        params.C = params.C - 1.*self.rate*params.grad/(np.sqrt(params.friction)+self.eps)
    def stepInput(self, input_node):
        input_node.incrementFrictions()
        for row in range(input_node.X_out.shape[0]):
            input_node.X_out[row] = input_node.X_out[row] - 1.*self.rate * \
                input_node.grad[row]/(np.sqrt(input_node.friction[row])+self.eps)

