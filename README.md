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
- MKL or GSL (MKL + Intel compilers are recommended!)
- Python packages: numpy, scipy, h5py, sklearn

### Notes
- The build process includes an installation of Boost 1.54.0 (components: python, serialization). If you would like to use a pre-installed version of Boost instead, simply set the corresponding build variable in build.sh accordingly:
  ```bash
  with_system_boost=false
  ```

