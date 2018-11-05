
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
