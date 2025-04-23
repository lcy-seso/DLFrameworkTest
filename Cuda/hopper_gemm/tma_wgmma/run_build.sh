#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

if [ -f "CMakeCache.txt" ]; then
    rm CMakeCache.txt
fi

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
fi

if [ -f "tma_wgmma" ]; then
    rm tma_wgmma
fi

cmake ..

make 2>&1 | tee ../build.log

if [ -f "tma_wgmma" ]; then
    echo "build success"
    ./tma_wgmma 2>&1 | tee ../run.log
else
    echo "build failed"
fi

cd ..
