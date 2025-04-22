#!/bin/bash

cd build

if [ -f "hopper_gemm" ]; then
    rm hopper_gemm
fi

if [ -f "CMakeCache.txt" ]; then
    rm CMakeCache.txt
fi

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
fi

cmake ..

make 2>&1 | tee ../build.log

if [ -f "hopper_gemm" ]; then
    echo "build success"
    ./hopper_gemm 2>&1 | tee ../run.log
else
    echo "build failed"
fi
cd ..
