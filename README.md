<div align="center">
    <img src="https://raw.githubusercontent.com/capoe/soapxx/master/web/media/soapxx.png" alt="logo"></img>
</div>

# soapxx

### Installation
```bash
git clone https://github.com/capoe/soapxx.git
cd soapxx
./build.sh
source soap/SOAPRC
```

### Dependencies
- Boost components: python, filesystem, serialization
- MKL or GSL (MKL + Intel compilers are recommended)
- Python packages: numpy, scipy, h5py, sklearn

### Notes
- boost::python needs to be compiled against the same Python version as used in the compilation (the latter being defined by the PYTHON_LIBRARY variable in build/CMakeCache.txt). To configure the Python version in the Boost installation, edit the project-config.jam configuration file of the Boost source directory accordingly. Note that Python3 is not presently supported.

