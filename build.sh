#! /bin/bash
with_system_boost=false

if [ "${with_system_boost}" = false ]; then
  echo "Installing boost ..."
  sleep 1.
  cd external
  #./build_boost.sh
  cd ..
fi

echo "Installing soap ..."
mkdir -p build
cd build
cmake .. -DWITH_SYSTEM_BOOST=${with_system_boost} && make -j 4 && make install
cd ..

