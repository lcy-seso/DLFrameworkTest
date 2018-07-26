#!/bin/bash

if [ ! -d data ]; then
  ln -s /data/data1/yincao/wmt14_English_German/exp_data data
fi

export CUDA_VISIBLE_DEVICES="7"
gdb -ex r --args python train.py \
    --batch_size=384 \
    --variable_update="replicated" \
    --independent_replica="true" \
    --use_synthetic_data="true" \
    --src_max_len=100 \
    --encoder_type="cudnn_lstm" \
    --direction="bi" \
