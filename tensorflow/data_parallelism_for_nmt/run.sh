#!/bin/bash

if [ ! -d data ]; then
  ln -s /data/data1/yincao/wmt14_English_German/exp_data data
fi

CUDA_VISIBLE_DEVICES="7" python train.py \
  --encoder_type="cudnn_lstm" \
  --batch_size=360 \
  --variable_update="replicated" \
  --gradient_repacking=4 \
  --all_reduce_spec="nccl" \
  --agg_small_grads_max_bytes=0 \
  --agg_small_grads_max_group=10 \
  #2>&1 | tee train.log
