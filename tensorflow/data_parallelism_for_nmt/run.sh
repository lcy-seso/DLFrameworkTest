#!/bin/bash

CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" python3 train.py \
  --batch_size=360 \
  --variable_update="replicated" \
  --gradient_repacking=4 \
  --all_reduce_spec="nccl" \
  --agg_small_grads_max_bytes=0 \
  --agg_small_grads_max_group=10 \
  #2>&1 | tee train.log
