#! /bin/bash
mkdir -p build
cd build
cmake .. -DCMAKE_CXX_COMPILER_ID=Intel && make && make install
cd ..
