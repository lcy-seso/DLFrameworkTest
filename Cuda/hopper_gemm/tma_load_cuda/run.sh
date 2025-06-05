#!/bin/bash

make 2>&1 | tee build.log

./build/tma_load_cuda 2>&1 | tee log.txt
