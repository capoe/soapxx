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

def as_tuple(parents):
    return (p.x for p in parents)

def as_list(parents):
    return ([ p.x for p in parents ],)

def check_torch():
    if torch.nn.Module is Mock:
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

identity = Identity()
concatenate = Concatenate()

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
    def setDependencies(self):
        self.deps = collections.OrderedDict()
        for p in self.parents:
            self.deps.update(p.deps)
        for p in self.parents:
            self.deps[p.tag] = p
        return self.deps
    def feed(self, x):
        self.x = x
    def printInfo(self, log=log, indent="  "):
        log << "%sNode %-3d  %-10s   #pars =" % (indent, self.idx, self.tag) << log.flush
        for p in self.module.parameters():
            log << "%7d" % np.prod(p.shape) << log.flush
        log << log.endl
    def randomizeParameters(self, cmin=-1.0, cmax=+1.0):
        for p in self.parameters():
            torch.nn.init.uniform_(p, a=cmin, b=cmax)
    def parameters(self):
        return self.module.parameters()
    def forward(self, log):
        if len(self.parents) > 0:
            args = self.format_args(self.parents)
            self.x = self.module.forward(*args, **self.kwargs)

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
        return self.node_map[self.tag+"."+node_tag]
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
    def feed(self, feed):
        for tag, x in feed.iteritems():
            if tag in self.node_map:
                self.node_map[tag].feed(x)
        for sub in self.feed_list: sub.feed(feed=feed)
    def forward(self, feed={}, node=None, lazy_set=set(), log=None, verbose=False):
        if len(feed) > 0: self.feed(feed=feed)
        if node is not None:
            path = [ n for nidx, n in node.deps.iteritems() ] + [ node ]
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
        
