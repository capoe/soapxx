#! /bin/bash
mkdir -p build
cd build
cmake .. && make && make install
#cmake .. -DCMAKE_CXX_COMPILER_ID=Intel && make && make install
cd ..
