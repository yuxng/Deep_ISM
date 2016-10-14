#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/shapenet_scene_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/shapenet_scene/solver.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb shapenet_scene_train \
  --cfg experiments/cfgs/shapenet_scene.yml \
  --iters 40000

#time ./tools/test_net.py --gpu $1 \
#  --def models/CaffeNet/shapenet_scene/test.prototxt \
#  --net output/shapenet_scene/shapenet_scene_train/caffenet_ism_shapenet_scene_iter_40000.caffemodel \
#  --imdb shapenet_scene_val \
#  --cfg experiments/cfgs/shapenet_scene.yml

#time ./tools/test_net.py --gpu $1 \
#  --def models/CaffeNet/shapenet_scene/test.prototxt \
#  --net output/shapenet_scene/shapenet_scene_train/caffenet_ism_shapenet_scene_iter_40000.caffemodel \
#  --imdb rgbd_scenes_val \
#  --cfg experiments/cfgs/shapenet_scene.yml
