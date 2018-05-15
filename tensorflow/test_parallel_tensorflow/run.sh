#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2" python train.py \
  2>&1 | tee train.log
