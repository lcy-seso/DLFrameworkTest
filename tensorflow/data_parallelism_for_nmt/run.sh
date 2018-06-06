#!/bin/bash

CUDA_VISIBLE_DEVICES="7" python3 train.py \
  --variable_update="replicated" \
  --gradient_repacking=4 \
  --all_reduce_spec="nccl" \
  --agg_small_grads_max_bytes=0 \
  --agg_small_grads_max_group=10 \
  2>&1 | tee train.log
