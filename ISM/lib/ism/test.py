# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from ism.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import math

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = []
    im_scale_factors = []
    scales = cfg.TEST.SCALES_BASE

    for im_scale in scales:
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)

    return blob, np.array(im_scale_factors)


def im_detect(net, im, num_classes):
    """Detect object classes in an image given boxes on grids.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of boxes

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    # compute image blob
    im_blob, im_scale_factors = _get_image_blob(im)

    # reshape network inputs
    net.blobs['data'].reshape(*(im_blob.shape))
    blobs_out = net.forward(data=im_blob.astype(np.float32, copy=False))

    # get outputs
    cls_prob = blobs_out['cls_prob']
    center_pred = blobs_out['center_pred']

    return cls_prob, center_pred


def vis_detections(im, cls_prob, center_pred):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    plt.cla()

    # show image
    plt.subplot(1, 3, 1)
    plt.imshow(im)

    # show class label
    label = cls_prob[0, 1, :, :]
    plt.subplot(1, 3, 2)
    plt.imshow(label)

    # show the target
    plt.subplot(1, 3, 3)
    plt.imshow(label)
    vx = center_pred[0, 0, :, :]
    vy = center_pred[0, 1, :, :]
    for x in xrange(vx.shape[1]):
        for y in xrange(vx.shape[0]):
            if vx[y, x] != 0 and vy[y, x] != 0:
                plt.gca().annotate("", xy=(x + vx[y, x], y + vy[y, x]), xycoords='data', xytext=(x, y), textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.show()


def test_net(net, imdb):

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    det_file = os.path.join(output_dir, 'detections.pkl')
    print imdb.name
    if os.path.exists(det_file):
        return

    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    detections = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        cls_prob, center_pred = im_detect(net, im, imdb.num_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()
        det = {'cls_prob': cls_prob, 'center_pred': center_pred}
        detections[i] = det
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time, _t['misc'].average_time)

        vis_detections(im, cls_prob, center_pred)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
