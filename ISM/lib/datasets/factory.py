# --------------------------------------------------------
# Deep ISM
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.shapenet
import datasets.shapenet_scene
import datasets.rgbd_scenes
import datasets.lov
import numpy as np

# shapenet dataset
for split in ['train', 'val']:
    name = 'shapenet_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.shapenet(split))

# shapenet dataset
for split in ['train', 'val']:
    name = 'shapenet_scene_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.shapenet_scene(split))

# rgbd scenes dataset
for split in ['val']:
    name = 'rgbd_scenes_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.rgbd_scenes(split))

# lov dataset
for split in ['train', 'val']:
    name = 'lov_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.lov(split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
