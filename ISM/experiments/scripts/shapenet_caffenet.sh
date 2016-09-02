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

#time ./tools/test_net.py --gpu $1 \
#  --def models/CaffeNet/shapenet/test.prototxt \
#  --net output/shapenet/shapenet_train/caffenet_ism_shapenet_iter_40000.caffemodel \
#  --imdb shapenet_val \
#  --cfg experiments/cfgs/shapenet.yml
