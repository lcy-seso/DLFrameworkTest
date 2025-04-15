#!/bin/bash

cd build

if [ -f "CMakeCache.txt" ]; then
    rm CMakeCache.txt
fi

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
fi

cmake ..

make 2>&1 | tee ../build.log

./hopper_gemm 2>&1 | tee ../run.log

cd ..
