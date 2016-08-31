#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/shapenet_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/shapenet/solver.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb shapenet_train \
  --cfg experiments/cfgs/shapenet.yml \
  --iters 40000
