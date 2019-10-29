import soap
import numpy as np
import time
import multiprocessing as mp
from . import nn
from . import momo
from . import tuplex
log = momo.osio

def load_extract_data(configs, basis, 
        with_nb_idcs=False, 
        max_envs_per_config=-1, 
        log=None):
    if log: log << log.mg << "Extract structures ..." << log.endl
    T_mat_list = []
    R_mat_list = []
    I_mat_list = []
    for cidx, config in enumerate(configs):
        if log: log << log.back << "Extracting structure %d" % cidx << log.flush
        T_mat, R_mat, I_mat = extract_environments(
            config=config, 
            basis=basis, 
            with_nb_idcs=with_nb_idcs,
            max_envs_per_config=max_envs_per_config)
        T_mat_list.append(T_mat)
        R_mat_list.append(R_mat)
        if with_nb_idcs: I_mat_list.append(I_mat)
    if log: log << log.endl
    if with_nb_idcs: return T_mat_list, R_mat_list, I_mat_list
    else: return T_mat_list, R_mat_list 

def extract_environments(config, basis, 
        with_nb_idcs=False, 
        max_envs_per_config=-1):
    R = config.positions
    T = np.array(config.symbols)
    R_out = []
    T_out = []
    I_out = []
    if "centres" in config.info:
        centres = map(int, config.info["centres"].split(','))
    else:
        centres = np.where(np.array(config.symbols) != 'H')[0]
    D = soap.tools.partition.calculate_distance_mat(R[centres], R)
    L = np.heaviside(basis.r_cut - D, 0.0)
    L[(np.arange(L.shape[0]), centres)] = 0
    for cc, c in enumerate(centres):
        nbs = np.where(L[cc] > 0.5)
        R_nbs = R[nbs]
        R_nbs = R_nbs - R[c]
        Rc = np.concatenate([ np.zeros((3,)), R_nbs.flatten() ])
        Tc = np.concatenate([ T[c:c+1], T[nbs] ])
        tids = ( np.arange(len(Tc)), [ basis.type_basis.encodeType(t) for t in Tc ] )
        Tc = np.zeros((len(Tc), basis.type_basis.N))
        Tc[tids] = 1.
        Tc = Tc.flatten()
        R_out.append(Rc)
        T_out.append(Tc)
        if with_nb_idcs: I_out.append(nbs)
    T_out = np.array(T_out)
    R_out = np.array(R_out)
    if with_nb_idcs: I_out = np.array(I_out)
    if max_envs_per_config > 0 and len(centres) > max_envs_per_config:
        sel = np.arange(len(centres))
        np.random.shuffle(sel)
        sel = sel[0:max_envs_per_config]
        T_out = T_out[sel]
        R_out = R_out[sel]
        if with_nb_idcs: I_out = I_out[sel]
    return T_out, R_out, I_out

def build_basis_default(type_contraction, type_set):
    if type_set == "default":
        types, pairs = soap.soapy.tuples.get_types_pairs_default(reduced=False)
    elif type_set == "cnosh":
        types, pairs = soap.soapy.tuples.get_types_pairs_cnosh()
    else: raise ValueError(type_set)
    type_basis = soap.soapy.tuples.TypeBasis(types, pairs)
    type_basis.setupPairContractionTensor(type_contraction)
    basis_x = soap.soapy.tuples.Basis(
        r_excl=0.9, 
        r_cut=4.0, 
        r_cut_width=0.0,
        sigma=0.5, 
        weight_decay_length=1.0,
        weight_constant=True,
        type_basis=type_basis, 
        r_err=0.1, 
        log=log)
    basis_f = soap.soapy.tuples.Basis(
        r_excl=0.9, 
        r_cut=4.0, 
        r_cut_width=0.0,
        sigma=0.5, 
        weight_decay_length=0.5,
        weight_constant=False,
        type_basis=type_basis, 
        r_err=0.1, 
        log=log)
    return basis_x, basis_f

class ConfigSubsampler(object):
    def __init__(self, 
            config_xyz_list,
            basis,
            max_envs_per_config,
            n_threads,
            rjit=0.1):
        self.configs = []
        for config_xyz in config_xyz_list:
            self.configs.extend(soap.tools.io.read(config_xyz))
        self.config_basis = basis
        self.n_threads = n_threads
        self.max_envs_per_config = max_envs_per_config
        self.rjit = rjit
    def sample(self, n_samples):
        config_idcs = np.random.randint(0, len(self.configs), size=(n_samples,))
        configs = [ self.configs[_] for _ in config_idcs ]
        T_configs, R_configs = load_extract_data(
            configs=configs, 
            basis=self.config_basis, 
            max_envs_per_config=self.max_envs_per_config)
        T_out = []
        R_out = []
        for i in range(len(R_configs)):
            for j in range(len(R_configs[i])):
                R_configs[i][j][3:] = R_configs[i][j][3:] + np.random.normal(0.0, self.rjit, size=(R_configs[i][j].shape[0]-3))
                T_out.append(T_configs[i][j])
                R_out.append(R_configs[i][j])
        T_configs = np.array(T_out)
        R_configs = np.array(R_out)
        return { "T": T_configs, "R": R_configs }
 
class TuplexPairSubsampler(object):
    def __init__(self, 
            lig_xyz, poc_xyz, dmats_npy, 
            lig_tuplex=None, poc_tuplex=None, 
            n_threads=1, rcut=4.5, rjit=0.1):
        self.configs_lig = soap.tools.io.read(lig_xyz)
        self.configs_poc = soap.tools.io.read(poc_xyz)
        # Contacts
        self.rcut = rcut
        self.dmats = np.load(dmats_npy)
        # Tuplexes
        if lig_tuplex is not None:
            self.lig_tuplex = lig_tuplex
            self.lig_tuplex["X"].w_grad = False
            self.lig_basis = lig_tuplex["X"].basis
            self.poc_tuplex = poc_tuplex
            self.poc_tuplex["X"].w_grad = False
            self.poc_basis = poc_tuplex["X"].basis
        self.n_threads = n_threads
        # Jitter
        self.rjit = rjit
        # Queue
        self.XA = []
        self.XB = []
        self.A = []
        self.B = []
    def expand_lmat(self, L):
        nA = L.shape[0]
        nB = L.shape[1]
        sel = np.where(L > 0.)
        idcs_a = sel[0]
        idcs_b = sel[1]
        yab = np.heaviside(self.rcut-L[sel].flatten(), 0.).reshape((-1,1))
        return { "a": idcs_a, "b": idcs_b, "y": yab }
    def precompute(self, n_batches, batch_size):
        self.XA = []
        self.XB = []
        self.A = []
        self.B = []
        args = []
        for n in range(n_batches):
            config_idcs = np.random.randint(0, len(self.configs_lig), size=(batch_size,))
            ligs = [ self.configs_lig[_] for _ in config_idcs ]
            pocs = [ self.configs_poc[_] for _ in config_idcs ]
            T_lig, R_lig = load_extract_data(ligs, self.lig_basis)
            T_poc, R_poc = load_extract_data(pocs, self.poc_basis)
            for i in range(len(R_lig)):
                for j in range(len(R_lig[i])):
                    R_lig[i][j][3:] = R_lig[i][j][3:] + np.random.normal(0.0, self.rjit, size=(R_lig[i][j].shape[0]-3))
            for i in range(len(R_poc)):
                for j in range(len(R_poc[i])):
                    R_poc[i][j][3:] = R_poc[i][j][3:] + np.random.normal(0.0, self.rjit, size=(R_poc[i][j].shape[0]-3))
            args.append([ self.lig_tuplex, self.poc_tuplex, T_lig, R_lig, T_poc, R_poc ])
            for b in range(batch_size):
                self.A.append(config_idcs[b])
                self.B.append(config_idcs[b])
        if self.n_threads > 1:
            pool = mp.Pool(processes=self.n_threads)
            XAXB_list = pool.map(evaluate_tuplex_lig_poc, args)
            pool.close()
        else:
            XAXB_list = map(evaluate_tuplex_lig_poc, args)
        for i in range(n_batches):
            for j in range(batch_size):
                self.XA.append(XAXB_list[i][0][j])
                self.XB.append(XAXB_list[i][1][j])
        return
    def sample(self, n_samples):
        if len(self.XA) < n_samples:
            t0 = time.time()
            self.precompute(n_batches=self.n_threads, batch_size=2*n_samples)
            t1 = time.time()
            log << "Precomputed %d samples: dt=%1.2f" % (self.n_threads*2*n_samples, (t1-t0)) << log.endl
        XA = []
        XB = []
        Y = []
        for s in range(n_samples):
            a = self.A.pop(0)
            b = self.B.pop(0)
            xa = self.XA.pop(0)
            xb = self.XB.pop(0)
            Dab = self.dmats[a]
            assert a == b
            assert Dab.shape[0] == xa.shape[0]
            assert Dab.shape[1] == xb.shape[0]
            contacts = self.expand_lmat(Dab)
            xa = xa[contacts["a"]]
            xb = xb[contacts["b"]]
            yab = contacts["y"]
            XA.append(xa)
            XB.append(xb)
            Y.append(yab)
        XA = np.concatenate(XA, axis=0)
        XB = np.concatenate(XB, axis=0)
        Y = np.concatenate(Y, axis=0)
        f = np.average(Y)
        log << "Batch: f = %+1.7e   xent(f) = %1.7e" % (f, -f*np.log(f)-(1-f)*np.log(1-f)) << log.endl
        return { "XA": XA, "XB": XB, "Y": Y}

def evaluate_tuplex(data):
    return data[0].evaluate(data[0]["X_out"], 
        feed={"T": data[1], "R": data[2]})

def evaluate_tuplex_lig_poc(data):
    # NOTE data = [ [ lig_tuplex, poc_tuplex, T_lig, R_lig, T_poc, R_poc ], ... ]
    assert len(data[2]) == len(data[3])
    assert len(data[2]) == len(data[4])
    assert len(data[2]) == len(data[5])
    XA = []
    XB = []
    for i in range(len(data[2])):
        if data[0] is not None:
            xa = data[0].evaluate(data[0]["X_out"], feed={"T": data[2][i], "R": data[3][i]})
            XA.append(xa)
        if data[1] is not None:
            xb = data[1].evaluate(data[1]["X_out"], feed={"T": data[4][i], "R": data[5][i]})
            XB.append(xb)
    return [XA, XB]

def build_tuplex_graph(args, basis_x, basis_f):
    g = nn.PyGraph()
    nT = g.addNode("flexinput", props={"tag": "T"}) # types
    nR = g.addNode("flexinput", props={"tag": "R"}) # coords
    nX = g.addNode("tuplex", parents=[nT,nR], props={
        "tag": "X", "basis": basis_x, "n_procs": args.n_procs})
    g.printInfo()
    return g

