#!/bin/bash

cd build

# if [ -f CMakeCache.txt ]; then
#   rm CMakeCache.txt
# fi

# if [ -d CMakeFiles ]; then
#   rm -r CMakeFiles
# fi

# cmake ../

make 2>&1 | tee ../build.log

cd ../

./build/pipelined_gemm 2>&1 | tee run.log
