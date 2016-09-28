#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/scenenet_val_vgg16_fcn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --network fcn8_vgg \
  --weights data/imagenet_models/vgg16.npy \
  --imdb scenenet_train \
  --cfg experiments/cfgs/scenenet_fcn.yml

time ./tools/test_net.py --gpu $1 \
  --network fcn8_vgg \
  --weights output/scenenet/scenenet_train/vgg16_fcn_scenenet_iter_40000.ckpt \
  --imdb scenenet_val \
  --cfg experiments/cfgs/scenenet_fcn.yml
