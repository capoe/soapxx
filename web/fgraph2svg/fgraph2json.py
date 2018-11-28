#! /usr/bin/env python
import pickle
import numpy as np
import sys
import os
import json
import soap
log = soap.soapy.momo.osio

log.Connect()
log.AddArg("fgraph", typ=str, help="CXX-FGraph archive file")
log.AddArg("npfga", typ=str, default="", help="NPFGA json-file")
options = log.Parse()

fgraph_cxx = soap.FGraph().load(options.fgraph)
fgraph = soap.soapy.npfga.PyFGraph(fgraph_cxx)
fgraph.rank(cumulative=True)
fgraph, curves, curve_info = soap.soapy.npfga.represent_graph_2d(fgraph)
root_nodes = filter(lambda f: f.is_root, fgraph.fnodes)

npfga = json.load(open(options.npfga))

nodes = {}
resolve = {}
for idx, fnode in enumerate(fgraph.fnodes):
    tag = "F%d" %  idx
    resolve[tag] = fnode.expr
    resolve[fnode.expr] = tag
    nodes[tag] = {
        "x": fnode.radius * np.cos(fnode.phi),
        "y": fnode.radius * np.sin(fnode.phi),
        "c": fnode.cov,
        "q": fnode.confidence,
        "id": tag,
        "expr": fnode.expr,
        "pars": [ p.expr for p in fnode.parents ],
    }
links = {}
for cinfo, curve in zip(curve_info, curves):
    tag_target = resolve[cinfo["target"]]
    if tag_target not in links: links[tag_target] = {}
    tag_source = resolve[cinfo["source"]]
    links[tag_target][tag_source] = { "xy": curve, "t": cinfo["target"], "s": cinfo["source"] }

fgraph_tag = os.path.basename(options.fgraph).replace('fgraph_cv_no_i000_iter0_', '').replace('.arch', '')

log >> 'mkdir -p fgraph2json'
json.dump(npfga["state"], open("fgraph2json/%s_state.json" % fgraph_tag, "w"), indent=1, sort_keys=True)
json.dump(nodes, open("fgraph2json/%s_nodes.json" % fgraph_tag, "w"), indent=1, sort_keys=True)
json.dump(links, open("fgraph2json/%s_links.json" % fgraph_tag, "w"), indent=1, sort_keys=True)
json.dump(resolve, open("fgraph2json/%s_resolve.json" % fgraph_tag, "w"), indent=1, sort_keys=True)

