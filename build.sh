#! /bin/bash
mkdir -p build
cd build
cmake .. \
-DCMAKE_CXX_COMPILER_ID=Intel \
-DPYTHON_INCLUDE_DIR=/usr/include/python2.6 \
-DPYTHON_LIBRARY=/usr/lib64/libpython2.6.so

make -j 4 && make install
cd ..
