# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Compute minibatch blobs for training a FCN network."""

import numpy as np
import numpy.random as npr
import cv2
from fcn.config import cfg
from utils.blob import im_list_to_blob
from utils.backprojection import backproject

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    im_blob, im_depth_blob, im_scales = _get_image_blob(roidb, random_scale_ind)

    # build the label blob
    label_blob = _get_label_blob(roidb, im_scales)

    # For debug visualizations
    _vis_minibatch(im_blob, im_depth_blob, label_blob)

    blobs = {'data_image': im_blob,
             'data_depth': im_depth_blob,
             'labels': label_blob}

    return blobs

def _get_image_blob(roidb, scale_ind):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ims_depth = []
    im_scales = []
    for i in xrange(num_images):
        # rgb
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scales.append(im_scale)
        processed_ims.append(im)

        # depth
        im_depth = cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED).astype(np.float32)

        # backprojection to compute normals
        im_depth = backproject(im_depth, roidb[i]['meta_data'])

        # im_depth = im_depth / im_depth.max() * 255
        # im_depth = np.tile(im_depth[:,:,np.newaxis], (1,1,3))
        if roidb[i]['flipped']:
            im_depth = im_depth[:, ::-1]

        im_orig = im_depth.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_depth = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_depth.append(im_depth)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, blob_depth, im_scales


def _get_label_blob(roidb, im_scales):
    """ build the label blob """

    num_images = len(roidb)
    processed_ims_cls = []

    for i in xrange(num_images):
        # read the label image
        im = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)
        if roidb[i]['flipped']:
            im = im[:, ::-1]

        im_cls = im.astype(np.float32, copy=True)

        # rescale image
        im_scale = im_scales[i]
        im = cv2.resize(im_cls, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        processed_ims_cls.append(im)

    # Create a blob to hold the input images
    blob_cls = im_list_to_blob(processed_ims_cls, 1)
    blob_shape = blob_cls.shape
    return blob_cls.reshape((blob_shape[0], blob_shape[1], blob_shape[2]))


def _vis_minibatch(im_blob, im_depth_blob, label_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    for i in xrange(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        fig.add_subplot(131)
        plt.imshow(im)

        # show depth image
        im_depth = im_depth_blob[i, :, :, :].copy()
        im_depth += cfg.PIXEL_MEANS
        im_depth = im_depth[:, :, (2, 1, 0)]
        im_depth = im_depth.astype(np.uint8)
        fig.add_subplot(132)
        plt.imshow(im_depth)

        # show label
        label = label_blob[i, :, :]
        fig.add_subplot(133)
        plt.imshow(label)

        plt.show()
