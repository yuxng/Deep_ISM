# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from ism.config import cfg
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
            # base_net = caffe.Net(pretrained_model+'.prototxt', pretrained_model+'.caffemodel', caffe.TEST)
            # self.transplant(self.solver.net, base_net)
            # del base_net

            for key, value in self.solver.net.params.iteritems():
                key_depth = key + '_d'
                if self.solver.net.params.has_key(key_depth):
                    self.solver.net.params[key_depth][0].data[...] = self.solver.net.params[key][0].data
                    self.solver.net.params[key_depth][1].data[...] = self.solver.net.params[key][1].data
                    print 'layer %s initialized from layer %s' % (key_depth, key)
        
        # surgeries
        interp_layers = [k for k in self.solver.net.params.keys() if 'up' in k]
        self.interp(self.solver.net, interp_layers)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def transplant(self, new_net, net, suffix=''):
        """
        Transfer weights by copying matching parameters, coercing parameters of
        incompatible shape, and dropping unmatched parameters.

        The coercion is useful to convert fully connected layers to their
        equivalent convolutional layers, since the weights are the same and only
        the shapes are different.  In particular, equivalent fully connected and
        convolution layers have shapes O x I and O x I x H x W respectively for O
        outputs channels, I input channels, H kernel height, and W kernel width.

        Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
        """

        for p in net.params:
            p_new = p + suffix
            if p_new not in new_net.params:
                print 'dropping', p
                continue
            for i in range(len(net.params[p])):
                if i > (len(new_net.params[p_new]) - 1):
                    print 'dropping', p, i
                    break
                if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                    print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
                else:
                    print 'copying', p, ' -> ', p_new, i
                new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

    def upsample_filt(self, size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

    def interp(self, net, layers):
        """
        Set weights of each layer in layers to bilinear kernels for interpolation.
        """
        for l in layers:
            m, k, h, w = net.params[l][0].data.shape
            if m != k and k != 1:
                print 'input + output channels need to be the same or |output| == 1'
                raise
            if h != w:
                print 'filters need to be square'
                raise
            filt = self.upsample_filt(h)
            net.params[l][0].data[range(m), range(k), :, :] = filt

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir, pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
