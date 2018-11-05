
### Workflow

Execute run.sh. This will create an output hdf5-file 'structures.xyz' containing the molecular graphs and molecular kernel matrix.
This NxN kernel matrix (where N is the number of structures) can be loaded using the h5py library:
```python
import h5py
f = h5py.File('data/structures.hdf5', 'r')
K = f['kernel']['kernel_mat'].value
```

The options for the atomic and molecular kernel are specified in kernel-compute.json (kernel type, exponent, basis set). 
This is the options file used by kernel-compute.py.
