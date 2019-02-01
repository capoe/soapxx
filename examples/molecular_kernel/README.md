
### Workflow

Execute run.sh. This will create an output hdf5-file 'structures.hdf5' containing the molecular graphs and molecular kernel matrix.

The options for the atomic and molecular kernel are specified in kernel-compute.json (kernel type, exponent, basis set). 
This is the options file used by kernel-compute.py.

The NxN kernel matrix (where N is the number of structures) can be loaded using the h5py library.
```python
import h5py
f = h5py.File('data/structures.hdf5', 'r')
K = f['kernel']['kernel_mat'].value
```

This is the matrix to be used in the regression of molecular properties. The rows and columns of this matrix correspond to the structures in the same order as listed in structures.xyz (this information can also be extracted from the information stored in the hdf5-file, see for example the path labels/label\_mat.
Note that raising the matrix to an even power xi=2 or 3 tends to improve performance.

For multiprocessing, adjust the command in run.sh to specify the number of cores and block size (a block size of, for example, 100 means that the kernel matrix is partitioned onto 100x100-sized chunks distributed over the individual cores):
```python
./kernel-compute.py ... ... ... --n_procs 16 --mp_kernel_block_size 100
```
For safety, make sure that you disable the in-built numpy parallelization used in certain matrix operations. To this end, export the following environment variables:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

