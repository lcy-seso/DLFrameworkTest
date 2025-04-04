#!/bin/bash

build_dir="_build"
if [ ! -d "$build_dir" ]; then
    mkdir $build_dir
fi

cd $build_dir

if [ -f "CMakeCache.txt" ]; then
    rm -rf CMakeCache.txt
fi

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
fi

cmake -DCMAKE_C_COMPILER=`which gcc` \
   -DCMAKE_CXX_COMPILER=`which g++` \
   .. 2>&1 | tee cmake.log

make -j 96 2>&1 | tee ../build.log

# ./_build/hopper_gemm

if [ -f "tma_load" ]; then
    echo "Run the executable"
    ./tma_load 2>&1 | tee ../run.log
else
    echo "Build failed"
fi

cd ../
