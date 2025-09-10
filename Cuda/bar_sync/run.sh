#!/bin/bash

TARGET="bar_sync"

cd build

if [ -f "CMakeCache.txt" ]; then
    rm -f CMakeCache.txt
fi

if [ -f $TARGET ]; then
    rm -f $TARGET
fi

cmake .. 2>&1 | tee ../build.log
make $TARGET -j 2>&1 | tee -a ../build.log

if [ -f $TARGET ]; then
    ./$TARGET 2>&1 | tee ../test.log
else
    echo "build $TARGET failed"
    exit 1
fi

cd ..
