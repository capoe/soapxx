#! /usr/bin/env python
import json
import numpy as np
import os
from . import ptable
from . import partition
from .. import _soapxx as soap

# ==============
# READ/FILTER IO
# ==============

class ConfigASE(object):
    def __init__(self):
        self.info = {}
        self.cell = None
        self.pbc = np.array([False, False, False])
        self.atoms = []
        self.positions = []
        self.symbols = []
    def __len__(self):
        return len(self.symbols)
    def get_positions(self):
        return self.positions
    def get_chemical_symbols(self):
        return self.symbols
    def get_atomic_numbers(self):
        return np.array([ ptable.lookup[s].z for s in self.get_chemical_symbols() ])
    def create(self, n_atoms, fs):
        #header = fs.readline().split()
        # Parse header: key1="str1" key2=123 key3="another value" ...
        header = fs.readline().replace("\n", "")
        tokens = []
        pos0 = 0
        pos1 = 0
        status = "<"
        quotcount = 0
        while pos1 < len(header):
            #print tokens, quotcount, status, pos0, pos1, header[pos0:pos1]
            status_out = status
            # On the lhs of the key-value pair?
            if status == "<":
                if header[pos1] == "=":
                    tokens.append(header[pos0:pos1])
                    pos0 = pos1+1
                    pos1 = pos1+1
                    status_out = ">"
                    quotcount = 0
                else:
                    pos1 += 1
            # On the rhs of the key-value pair?
            elif status == ">":
                if header[pos1-1:pos1] == '"':
                    quotcount += 1
                if quotcount == 0 and header[pos1] == ' ':
                    quotcount = 2
                if quotcount <= 1:
                    pos1 += 1
                elif quotcount == 2:
                    tokens.append(header[pos0:pos1])
                    pos0 = pos1+1
                    pos1 = pos1+1
                    status_out = ""
                    quotcount = 0
                else:
                    assert False
            # In between key-value pairs?
            elif status == "":
                if header[pos1] == ' ':
                    pos0 += 1
                    pos1 += 1
                else:
                    status_out = "<"
            else:
                assert False
            status = status_out
        kvs = []
        for i in range(len(tokens)/2):
            kvs.append([tokens[2*i], tokens[2*i+1]])
        # Process key-value pairs
        for kv in kvs:
            key = kv[0]
            value = '='.join(kv[1:])
            value = value.replace('"','').replace('\'','')
            # Float?
            if '.' in value:
                try:
                    value = float(value)
                except: pass
            else:
                # Int?
                try:
                    value = int(value)
                except: pass
            self.info[kv[0]] = value
        # Read atoms
        self.positions = []
        self.symbols = []
        for i in range(n_atoms):
            new_atom = self.create_atom(fs.readline())
            self.positions.append(new_atom.pos)
            self.symbols.append(new_atom.name)
        self.positions = np.array(self.positions)
        return
    def create_atom(self, ln):
        ln = ln.split()
        name = ln[0]
        pos = map(float, ln[1:4])
        pos = np.array(pos)
        new_atom = AtomASE(name, pos)
        self.atoms.append(new_atom)
        return new_atom

class AtomASE(object):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

class IO(object):
    def __init__(self):
        self.ConfigASE = ConfigASE
        return
    def convert(self, config, tag="?", sigma=0.5, weight=None, typemap=None, laplace_cutoff=0):
        R = config.get_positions()
        T = config.get_chemical_symbols()
        N = R.shape[0]
        if weight is None: weight = np.ones((N,))
        if type(sigma) in [float, np.float64, np.float32]:
            sigma = sigma*np.ones((N,))
        if config.pbc.all(): 
            box = np.array([config.cell[0], config.cell[1], config.cell[2]])
        elif not config.pbc.any(): 
            box = np.zeros((3,3))
        else: 
            raise NotImplementedError("<IO::convert> Partial periodicity not implemented.")
        struct = soap.Structure(tag)
        struct.box = box
        segment = struct.addSegment()
        for i in range(R.shape[0]):
            r = R[i]
            t = T[i]
            particle = struct.addParticle(segment)
            particle.pos = r
            particle.weight = weight[i]
            particle.sigma = sigma[i]
            if typemap is None: particle.type = T[i]
            else:
                colour = typemap[T[i]]
                for channel_idx, c in enumerate(colour):
                    particle.addType(typemap["channels"][channel_idx], c)
        if laplace_cutoff > 0:
            D = partition.calculate_distance_mat(R, R)
            L = 1*partition.calculate_connectivity_mat(D, T)
            np.fill_diagonal(L, 0)
            L_out = np.copy(L)
            L_visited = np.copy(L)
            for lcut in range(2, laplace_cutoff+1):
                dL_visited = np.heaviside(np.heaviside(L_visited.dot(L),0) - L_visited, 0)
                np.fill_diagonal(dL_visited, 0)
                L_out = L_out + lcut*dL_visited
                L_visited = L_visited + dL_visited
            L_out = L_out + 2*laplace_cutoff*(1-L_visited)
            np.fill_diagonal(L_out, 0)
            struct.setLaplacian(L_out, str(L_out.dtype))
        return struct
    def read(
            self,
            config_file,
            index=':'):
        configs = []
        ifs = open(config_file, 'r')
        while True:
            header = ifs.readline().split()
            if header != []:
                assert len(header) == 1
                n_atoms = int(header[0])
                config = ConfigASE()
                config.create(n_atoms, ifs)
                configs.append(config)
            else: break
        return configs
    def write(
            self,
            config_file,
            configs):
        if type(configs) != list:
            configs = [ configs ]
        ofs = open(config_file, 'w')
        for c in configs:
            ofs.write('%d\n' % (len(c)))
            for k in sorted(c.info.keys()):
                # int or float?
                if type(c.info[k]) not in [ unicode, str ]:
                    ofs.write('%s=%s ' % (k, c.info[k]))
                # String
                else:
                    ofs.write('%s="%s" ' % (k, c.info[k]))
            ofs.write('\n')
            for i in range(len(c)):
                ofs.write('%s %+1.4f %+1.4f %+1.4f\n' % (
                    c.get_chemical_symbols()[i], c.positions[i][0], c.positions[i][1], c.positions[i][2]))
        ofs.close()
        return

def read_filter_configs(
        config_file, 
        index=':', 
        filter_types=None, 
        types=[],
        do_remove_duplicates=False, 
        key=lambda c: c.info['label'],
        log=None):
    if log: log << "Reading" << config_file << log.endl
    configs = read_xyz(config_file, index=index)
    if log: log << log.item << "Have %d initial configurations" % len(configs) << log.endl
    if do_remove_duplicates:
        configs, duplics = remove_duplicates(configs, key=key)
        if log: log << log.item << "Removed %d duplicates" % len(duplics) << log.endl
    if filter_types:
        configs_filtered = []
        for config in configs:
            types_config = config.get_chemical_symbols()
            keep = True
            for t in types_config:
                if not t in types:
                    keep = False
                    break
            if keep: configs_filtered.append(config)
        configs = configs_filtered
        if log: log << log.item << "Have %d configurations after filtering" % len(configs) << log.endl
    return configs

def remove_duplicates(array, key=lambda a: a):
    len_in = len(array)
    label = {}
    array_curated = []
    array_duplicates = []
    for a in array:
        key_a = key(a)
        if key_a in label:
            array_duplicates.append(a)
        else:
            array_curated.append(a)
            label[key_a] = True
    len_out = len(array_curated)
    return array_curated, array_duplicates

io = IO()
