#!/bin/bash

modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
# sudo /Data/yincao/cuda-11.6/bin/ncu \
#   -c 10 -s 3 \
#   --section SpeedOfLight \
#   --section MemoryWorkloadAnalysis \
#   --section LaunchStats \
#   -o CopyTest -f load_test

# sudo /Data/yincao/cuda-11.6/bin/ncu \
#   --section "regex:MemoryWorkloadAnalysis(_Chart|_Tables)?" load_test 2>&1 | tee profiler.log
sudo /Data/yincao/cuda-11.6/bin/ncu \
  --metrics "dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed" load_test
