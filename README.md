<div align="center">
    <img src="https://raw.githubusercontent.com/capoe/soapxx/master/web/media/soapxx.png" alt="logo"></img>
</div>

# soapxx

### Installation
```bash
git clone https://github.com/capoe/soapxx.git
cd soapxx
./build.sh --install_boost
source soap/SOAPRC
```

### Dependencies
- MKL (+ Intel compilers recommended)
- Python packages: numpy, scipy, h5py

### Notes
If build.sh is executed with the --install_boost option, the build process includes a partial local installation of Boost 1.54.0 (components: python, serialization). You can omit this option if you already have Boost version < 1.67 installed on your system. Subsequent builds should be run without this option, unless reinstalling Boost is explicitly desired (e.g., due to a compiler change).

