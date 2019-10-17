from _soapxx import *
from linalg import *
from . import soapy
from . import tools
log = soapy.momo.osio

# Threading options
import ctypes
try: MKL_RT = ctypes.CDLL('libmkl_rt.so')
except OSError: MKL_RT = None

def mkl_set_num_threads(cores):
    if MKL_RT is None: raise OSError("set_num_threads only available with MKL")
    MKL_RT.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

# Objects emulating CXX entities
from soapy.dmap import DMapMatrixSuperSet, SuperKernel

