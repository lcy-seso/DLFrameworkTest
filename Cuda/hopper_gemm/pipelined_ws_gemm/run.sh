#!/bin/bash

cd build

exe_name="pipelined_gemm"

if [ -f $exe_name ]; then
  rm $exe_name
fi

if [ -f CMakeCache.txt ]; then
  rm CMakeCache.txt
fi

if [ -d CMakeFiles ]; then
  rm -r CMakeFiles
fi

cmake ../

make 2>&1 | tee ../build.log

if [ -f $exe_name ]; then
  ./$exe_name 2>&1 | tee ../run.log
else
  echo "build failed."
fi

cd ../
