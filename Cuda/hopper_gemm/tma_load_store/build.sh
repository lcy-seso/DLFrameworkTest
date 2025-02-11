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

cmake ..
make -j96 2>&1 | tee ../build.log

cd ../
