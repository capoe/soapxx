#! /usr/bin/env python
import os
import sys
import glob
import json

js = glob.glob('*.json')
tags = sorted(glob.glob('*_nodes.json'))
tags = [ t[:-11] for t in tags ]
includes = []
for t in tags:
    nodes = json.load(open('%s_nodes.json' % t))
    includes.append({ "tag": t, "n_features": len(nodes) })
json.dump(includes, open("include.json", "w"), indent=1, sort_keys=True)

