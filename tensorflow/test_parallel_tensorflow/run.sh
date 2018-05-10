#!/bin/bash

CUDA_VISIBLE_DEVICES="7" python train.py \
  2>&1 | tee train.log
