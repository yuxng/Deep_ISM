#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/lov_vgg16_color.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/lov/solver_color.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb lov_train \
  --cfg experiments/cfgs/lov.yml \
  --iters 40000

if [ -f $PWD/output/lov/lov_val/vgg16_fcn_color_lov_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/lov/lov_val/vgg16_fcn_color_lov_iter_40000/segmentations.pkl
fi

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/lov/test_color.prototxt \
  --net output/lov/lov_train/vgg16_fcn_color_lov_iter_40000.caffemodel \
  --imdb lov_val \
  --cfg experiments/cfgs/lov.yml
