#! /usr/bin/env python
import pickle
import numpy as np
import sys
import json
import soap

fgraph_cxx = soap.FGraph().load(sys.argv[1])
fgraph = soap.soapy.npfga.PyFGraph(fgraph_cxx)
fgraph, curves = soap.soapy.npfga.represent_graph_2d(fgraph)
root_nodes = filter(lambda f: f.is_root, fgraph.fnodes)

root_weights = json.load(open(sys.argv[2]))[
    "cv_instances"][0][
        "iterations"][0][
            "npfga_decomposition"][
                "root_weights"][
                    "root_weights"]

ofs = open('out_root.tab', 'w')
for fnode in root_nodes:
    x = fnode.radius * np.cos(fnode.phi)
    y = fnode.radius * np.sin(fnode.phi)
    ofs.write('%+1.7f %+1.7f %+1.7f %+1.7f %s\n' % (
        x, y, np.abs(fnode.cov*fnode.confidence), root_weights[fnode.expr], fnode.expr))
ofs.close()

ofs = open('out_root_weights.tab', 'w')
for fnode in root_nodes:
    x = fnode.radius * np.cos(fnode.phi)
    y = fnode.radius * np.sin(fnode.phi)
    ofs.write('%+1.7f %+1.7f 0.1 %s\n' % (0.1*x, 0.1*y, fnode.expr))
    ofs.write('%+1.7f %+1.7f 0.1 %s\n' % (1.0*x, 1.0*y, fnode.expr))
    ofs.write('\n')
ofs.close()

ofs = open('out_nodes.tab', 'w')
for fnode in fgraph.fnodes:
    x = fnode.radius * np.cos(fnode.phi)
    y = fnode.radius * np.sin(fnode.phi)
    ofs.write('%+1.7f %+1.7f %+1.7e %d %s\n' % (
        x, y, np.abs(fnode.cov*fnode.confidence), len(fnode.parents), fnode.expr))
ofs.close()

ofs = open('out_curves.tab', 'w')
for curve in curves:
    for pt in curve:
        ofs.write('%+1.4f %+1.4f %+1.4f\n' % tuple(pt))
    ofs.write('\n')
ofs.close()

