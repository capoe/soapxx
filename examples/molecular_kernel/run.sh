#! /bin/bash
./kernel-compute.py \
 -f data \
 -c structures.xyz \
 -l tag \
 -o kernel-compute.json \
 -s -1 \
 -hd structures.hdf5 \
 -t true

