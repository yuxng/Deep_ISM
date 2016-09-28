# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

import sys
import networks.fcn8_vgg
import tensorflow as tf

def get_network(name, pretrained_model):
    """Get a network by name."""
    if name == 'fcn8_vgg':
        return networks.FCN8VGG(pretrained_model)
    else:
        print 'network `{:s}` is not supported'.format(name)
        sys.exit()
