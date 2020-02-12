from _soapxx import *
from linalg import *
from . import soapy
from . import tools
from . import external
log = soapy.momo.osio
Config = tools.ase_io.ConfigASE

# Threading options
import ctypes
try: MKL_RT = ctypes.CDLL('libmkl_rt.so')
except OSError: MKL_RT = None

def mkl_set_num_threads(cores):
    if MKL_RT is None: raise OSError("set_num_threads only available with MKL")
    MKL_RT.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

class mkl_single_threaded():
    def __init__(self, n_threads_restore):
        self.n_threads = n_threads_restore
    def __enter__(self):
        mkl_set_num_threads(1)
    def __exit__(self, type, value, traceback):
        mkl_set_num_threads(self.n_threads)

# Objects emulating CXX entities
from soapy.dmap import DMapMatrixSuperSet, SuperKernel
from soapy import Args

