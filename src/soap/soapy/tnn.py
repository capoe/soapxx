import numpy as np
import collections
import pickle
from momo import log, Mock
try:
    import torch
    import torch.nn
except ImportError:
    torch = Mock()
    torch.nn = Mock()
    torch.nn.Module = Mock
    Mock.double = (lambda self: self)
    Mock.parameters = (lambda self: [])

def as_tuple(parents):
    return (p.x for p in parents)

def as_list(parents):
    return ([ p.x for p in parents ],)

def check_torch():
    if torch.nn.Module is Mock:
        log << log.mr << "WARNING torch not found" << log.endl
        raise ImportError("torch not found")

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Concatenate(torch.nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()
    def forward(self, *args, **kwargs):
        return torch.cat(*args, **kwargs)

class Reshape(torch.nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, *args, **kwargs):
        return torch.reshape(*args, **kwargs)

class Mult(torch.nn.Module):
    def __init__(self):
        super(Mult, self).__init__()
    def forward(self, *args, **kwargs):
        assert len(args) == 2
        return args[0]*args[1]

class Add(torch.nn.Module):
    def __init__(self, coeffs=[1.,1.]):
        super(Add, self).__init__()
        self.coeffs = coeffs
    def forward(self, *args, **kwargs):
        assert len(args) == 2
        return self.coeffs[0]*args[0]+self.coeffs[1]*args[1]

class MovingTarget(torch.nn.Module):
    def __init__(self, obj_1, obj_2, prob_1):
        super(MovingTarget, self).__init__()
        self.obj_1 = obj_1
        self.obj_2 = obj_2
        self.prob_1 = prob_1
        self.current = self.obj_1
    def forward(self, *args, **kwargs):
        if np.random.uniform() <= self.prob_1:
            self.current = self.obj_1
            return self.obj_1(*args, **kwargs)
        else:
            self.current = self.obj_2
            return self.obj_2(*args, **kwargs)
    def __repr__(self):
        return str(self.obj_1)+" : "+str(self.obj_2)+" = %1.1f : %1.1f" % (
            self.prob_1, 1.-self.prob_1)

identity = Identity()
concatenate = Concatenate()
reshape = Reshape()
mult = Mult()
add = Add()

class ModuleNode(object):
    def __init__(self, idx=-1, tag="", 
            module=identity, 
            parents=[],
            format_args=as_tuple,
            kwargs={}):
        self.idx = idx
        self.tag = tag
        self.module = module.double()
        self.parents = parents
        self.x = None
        self.deps = collections.OrderedDict()
        self.format_args = format_args
        self.requires_feed = False
        self.kwargs = kwargs
        self.requires_grad_ = True
    def setDependencies(self):
        self.deps = collections.OrderedDict()
        for p in self.parents:
            self.deps.update(p.deps)
        for p in self.parents:
            self.deps[p.tag] = p
        return self.deps
    def parameters(self):
        return self.module.parameters()
    def nParamSets(self):
        return len([ _ for _ in self.parameters() ])
    def requiresGrad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad
        self.requires_grad_ = requires_grad
    def randomizeParameters(self, cmin=-1.0, cmax=+1.0):
        for p in self.parameters():
            torch.nn.init.uniform_(p, a=cmin, b=cmax)
    def zeroParameters(self):
        for p in self.parameters():
            torch.nn.init.constant_(p, 0.0)
    def feed(self, x):
        self.x = x
    def forward(self, log):
        if len(self.parents) > 0:
            args = self.format_args(self.parents)
            self.x = self.module.forward(*args, **self.kwargs)
    def printInfo(self, log=log, indent="  "):
        log << "%sNode %-3d  %-10s   " % (indent, self.idx, self.tag) << log.flush
        log << "%-75s  " % repr(self.module).replace(
            "\n","").replace("  "," ").replace("_features","") << log.flush
        log << "<-"+"%-20s" % (",".join([ p.tag.split(".")[1] for p in self.parents ])) << log.flush
        log << "<-" << log.flush
        for p in self.module.parameters():
            log << "%-7d" % np.prod(p.shape) << log.flush
        if not self.requires_grad_: log << " [no grad]" << log.flush
        log << log.endl
    def printRange(self, log=log, indent="  "):
        log << "%sNode %-3d  %-10s   " % (indent, self.idx, self.tag) << log.endl
        for p in self.module.parameters():
            log << "%s  |c|=%-7d   min avg std max = %+1.4e %+1.4e %+1.4e %+1.4e" % (
                indent, np.prod(p.shape), torch.min(p), torch.mean(p), 
                torch.std(p), torch.max(p)) << log.endl

class ModuleGraph(ModuleNode):
    def __init__(self, idx=-1, tag="", module=identity, parents=[], 
            format_args=as_tuple, kwargs={}):
        check_torch()
        ModuleNode.__init__(self, 
            idx=idx, 
            tag=tag, 
            module=module, 
            parents=parents, 
            format_args=format_args,
            kwargs=kwargs)
        self.nodes = []
        self.node_map = {}
        self.dependency_map = {}
        self.requires_feed = True
        self.feed_list = []
    def __getitem__(self, node_tag):
        retag = "%s.%s" % (self.tag, node_tag) if not "." in node_tag else node_tag
        return self.node_map[retag]
    def create(self, tag, parents=[], module=identity, 
            format_args=as_tuple, kwargs={}):
        idx = len(self.nodes)
        if tag == "": tag = "%d" % idx
        new_node = ModuleNode(idx=idx, tag=self.tag+"."+tag, 
            parents=parents, module=module, format_args=format_args, kwargs=kwargs)
        new_node.setDependencies()
        self.register(new_node)
        return new_node
    def register(self, node):
        if node.requires_feed: self.feed_list.append(node)
        if node.tag != "":
            if node.tag in self.node_map: 
                raise ValueError("Node with identical tag '%s'" \
                    "already registered" % node.tag)
            self.node_map[node.tag] = node
        self.nodes.append(node)
        return node
    def updateDependencies(self):
        for node in self.nodes:
            node.setDependencies()
    def parameters(self):
        for n in self.nodes:
            for p in n.parameters():
                yield p
    def randomizeParameters(self, cmin=-1.0, cmax=1.0):
        for n in self.nodes: n.randomizeParameters(cmin, cmax)
    def importParameters(self, other, assignment={}):
        # assignment: { "other.node1": "this.node2", "other.node2": "this.node0", ... }
        log << "Import parameters" << log.endl
        if len(assignment) > 0:
            nodes_other = filter(lambda n: n.tag in assignment, other.nodes)
        else: nodes_other = other.nodes
        for node in nodes_other:
            tag_this = assignment[node.tag] if len(assignment) else node.tag
            node_this = self.node_map[tag_this]
            params_this = [ p for p in node_this.parameters() ]
            params_other = [ p for p in node.parameters() ]
            assert len(params_this) == len(params_other)
            log << "  Copying %d parameter sets, node '%s' -> '%s' [" % (
                len(params_this), node.tag, node_this.tag) << log.flush
            for i in range(len(params_this)):
                log << params_this[i].shape << log.flush
                assert params_this[i].shape == params_other[i].shape
                params_this[i].data = params_other[i].data
            log << "]" << log.endl
    def printInfo(self, indent=""):
        n_par = 0
        n_par_const = 0
        n_par_sets = 0
        for p in self.parameters():
            n_par_sets += 1
            n_par += np.prod(p.shape)
            if not p.requires_grad: n_par_const += np.prod(p.shape)
        log << "%sGraph %s with %d nodes and %d parameter sets with %d " \
            " parameters (of which %d constant)" % (
            indent, self.tag, len(self.nodes), n_par_sets, n_par, n_par_const) << log.endl
        for node in self.nodes:
            node.printInfo(indent=indent+"  ")
    def printRange(self, indent=""):
        log << "%sGraph %s: Parameter ranges" % (indent, self.tag) << log.endl
        for node in self.nodes:
            node.printRange(indent=indent+"  ")
    def feed(self, feed):
        for tag, x in feed.iteritems():
            if tag in self.node_map:
                self.node_map[tag].feed(x)
            else:
                log << "WARNING '%s' not found" % tag << log.endl
        for sub in self.feed_list: sub.feed(feed=feed)
    def forward(self, feed={}, endpoint=None, lazy_set=set(), log=None, verbose=False):
        if len(feed) > 0: self.feed(feed=feed)
        if endpoint is not None:
            path = [ n for nidx, n in endpoint.deps.iteritems() ] + [ endpoint ]
        else: path = self.nodes
        if verbose: log << "%s::forward [" % self.tag << log.flush
        for pathnode in path:
            if verbose: log << "->" << pathnode.tag << log.flush
            if pathnode.tag in lazy_set: continue
            pathnode.forward(log=log)
        if verbose: log << "]" << log.endl
        self.x = path[-1].x
        return self.x
    def save(self, archfile):
        with open(archfile, 'w') as f:
            f.write(pickle.dumps(self))
    def load(self, archfile):
        self = pickle.load(open(archfile, 'rb')) 
        return self 
        
