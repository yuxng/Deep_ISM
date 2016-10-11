# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an imdb (image database)."""

from fcn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import cPickle
from utils.blob import im_list_to_blob
from utils.backprojection import backproject
import os
import math
import tensorflow as tf

def _get_image_blob(im, im_depth, meta_data):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    # RGB
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = []
    im_scale_factors = []
    assert len(cfg.TEST.SCALES_BASE) == 1
    im_scale = cfg.TEST.SCALES_BASE[0]

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # depth
    # backprojection to compute normals
    im_depth = backproject(im_depth, meta_data)
    im_orig = im_depth.astype(np.float32, copy=True)

    # im_orig = im_orig / im_orig.max() * 255
    # im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_depth = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_depth.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, blob_depth, np.array(im_scale_factors)


def im_segment(sess, net, im, im_depth, meta_data, num_classes):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    # compute image blob
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)

    # forward pass
    feed_dict={net.data: im_depth_blob}
    pred_up = sess.run([net.pred_up], feed_dict=feed_dict)

    labels = pred_up[0]
    labels_shape = labels.shape

    return labels.reshape((labels_shape[1], labels_shape[2]))


def vis_segmentations(im, im_depth, labels):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    # show image
    fig.add_subplot(131)
    plt.imshow(im)

    # show depth image
    fig.add_subplot(132)
    plt.imshow(im_depth)

    # show label
    fig.add_subplot(133)
    plt.imshow(labels)

    plt.show()

def test_net(sess, net, imdb, weights_filename):

    output_dir = get_output_dir(imdb, weights_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    print imdb.name
    if os.path.exists(seg_file):
        with open(seg_file, 'rb') as fid:
            segmentations = cPickle.load(fid)
        imdb.evaluate_segmentations(segmentations, output_dir)
        return

    """Test a FCN on an image database."""
    num_images = len(imdb.image_index)
    segmentations = [[] for _ in xrange(num_images)]

    roidb = imdb.roidb

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    perm = np.random.permutation(np.arange(num_images))

    # for i in xrange(num_images):
    for i in perm:
        im = cv2.imread(roidb[i]['image'])
        im_depth = cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED)
        meta_data = roidb[i]['meta_data']

        _t['im_segment'].tic()
        labels = im_segment(sess, net, im, im_depth, meta_data, imdb.num_classes)
        _t['im_segment'].toc()

        _t['misc'].tic()
        seg = {'labels': labels}
        segmentations[i] = seg
        _t['misc'].toc()

        vis_segmentations(im, im_depth, labels)
        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].average_time, _t['misc'].average_time)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)
