#! /bin/bash
# Download and extract boost
if [ ! -f boost_1_54_0.tar.bz2 ]; then
    wget https://sourceforge.net/projects/boost/files/boost/1.54.0/boost_1_54_0.tar.bz2/download -O boost_1_54_0.tar.bz2
fi
tar -xvf boost_1_54_0.tar.bz2
# Copy required files: File list obtained using
# $ mkdir -p boost; ./dist/bin/bcp boost/tokenizer.hpp boost/archive/binary_iarchive.hpp boost/archive/binary_oarchive.hpp boost/archive/text_iarchive.hpp boost/archive/text_oarchive.hpp boost/format.hpp boost/lexical_cast.hpp boost/math/special_functions/erf.hpp boost/math/special_functions/legendre.hpp boost/math/special_functions/spherical_harmonic.hpp boost/numeric/ublas/io.hpp boost/numeric/ublas/lu.hpp boost/numeric/ublas/vector.hpp boost/numeric/ublas/matrix.hpp boost/numeric/ublas/symmetric.hpp boost/python.hpp boost/python/iterator.hpp boost/python/numeric.hpp boost/python/suite/indexing/vector_indexing_suite.hpp boost/serialization/base_object.hpp boost/serialization/complex.hpp boost/serialization/export.hpp boost/serialization/list.hpp boost/serialization/map.hpp boost/serialization/vector.hpp build bootstrap.bat bootstrap.sh boostcpp.jam boost-build.jam boost
rm -rf boost
mkdir -p boost
cd boost_1_54_0
cp --parents $(cat ../deps.txt | xargs) ../boost/.
cd ..
# Build from source
cd boost
rm -rf bin.v2
rm -f bootstrap.log b2 bjam project-config.jam*
./bootstrap.sh
./b2 install --prefix="../../soap"
cd ..
# Clean up
rm -r boost_1_54_0

