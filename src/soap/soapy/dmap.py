import soap
import numpy as np
import momo
log = momo.osio

class DMapMatrixSuperSet(object):
    """
    Represents structure ensembles
    Hierarchy: DMap -> DMapMatrix -> DMapMatrixSet -> DMapMatrixSuperSet
    """
    def __init__(self, dset, slices):
        self.dset = dset
        self.dmap_ensembles = []
        self.slices = slices
        for dset_slice in slices:
            ens = soap.DMapMatrixSet()
            for s in dset_slice: ens.append(dset[s])
            self.dmap_ensembles.append(ens)
        return
    def __getitem__(self, idx):
        return self.dmap_ensembles[idx]
    def __len__(self):
        return len(self.dmap_ensembles)

class SuperKernel(object):
    """
    Kernel calculator for DMapMatrixSuperSets
    Hierarchy: Basekernel -> Topkernel -> Superkernel -> Hyperkernel
    """
    def __init__(self, kernel_options_top, kernel_options_super):
        self.kernel_top = soap.Kernel(kernel_options_top)
        self.kernel_super = soap.Kernel(kernel_options_super)
        return
    def evaluate(self, d1, d2, symmetric=None, dtype_str="float64"):
        if type(d1) is soap.DMapMatrixSet and type(d2) is soap.DMapMatrixSet:
            k_top = self.kernel_top.evaluate(d1, d2, False, dtype_str)
            k_sup = self.kernel_super.evaluateTop(k_top, dtype_str)
            return k_sup
        elif type(d1) is DMapMatrixSuperSet and type(d2) is DMapMatrixSuperSet:
            assert symmetric != None and dtype_str != None
            if symmetric:
                K_top = self.kernel_top.evaluate(d1.dset, d2.dset, symmetric, dtype_str)
                diag = np.copy(K_top.diagonal())
                K_top = K_top + K_top.T
                np.fill_diagonal(K_top, diag)
                log << "Top kernel =" << K_top.shape << log.endl
                log << K_top << log.endl
                K_super = np.zeros((len(d1),len(d2)), dtype=K_top.dtype)
                for i in range(len(d1)):
                    slice_i = d1.slices[i]
                    log << log.back << "Row %d/%d" % (i,len(d1)) << log.flush
                    for j in range(i, len(d2)):
                        slice_j = d2.slices[j]
                        kij = self.kernel_super.evaluateTop(K_top[slice_i][:,slice_j], dtype_str)
                        K_super[i,j] = kij
                        K_super[j,i] = kij
                log << log.endl
                log << "Super kernel =" << K_super.shape << log.endl
                log << K_super << log.endl
            else:
                assert False # Not implemented
            return K_top, K_super
        elif type(d1) is soap.DMapMatrixSet and type(d2) == DMapMatrixSuperSet:
            K = np.zeros((len(d2),), dtype_str)  
            for i in range(len(d2)):
                K[i] = self.evaluate(d1, d2[i])
            return K
        else:
            assert False
    def attributeLeft(self, dmap_ensemble_probe, dmap_ensemble_set, dtype_str):
        degree = len(dmap_ensemble_probe)
        # Evaluate K_Aa|B
        K_top_attr = []
        for i in range(degree):
            K_top_attr_i = self.kernel_top.attributeLeft(
                dmap_ensemble_probe[i],
                dmap_ensemble_set.dset,
                dtype_str)
            K_top_attr.append(K_top_attr_i)
        # Evaluate P_{A}{B}
        K_top = self.kernel_top.evaluate(
            dmap_ensemble_probe,
            dmap_ensemble_set.dset,
            False,
            dtype_str)
        P_top = []
        for B in dmap_ensemble_set.slices:
            KAB = K_top[:,B]
            PAB = self.kernel_super.attributeTopGetReductionMatrix(
                KAB, dtype_str)
            P_top.append(PAB)
        # Evaluate K_Aa|{B} = sum_{B} sum_B K_Aa|B
        K_sup_attr = []
        for A in range(degree):
            KA_attr = []
            for i, B in enumerate(dmap_ensemble_set.slices):
                KaB = K_top_attr[A][:,B]
                PaAB = P_top[i][A]
                KA_attr.append(np.sum(KaB*PaAB, axis=1).reshape((-1,1)))
            KA_attr = np.concatenate(KA_attr, axis=1)
            K_sup_attr.append(KA_attr)
        K_sup_attr = np.concatenate(K_sup_attr, axis=0) 
        return K_sup_attr

