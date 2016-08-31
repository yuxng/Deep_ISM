# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from ism.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob = _get_image_blob(roidb)

    # build the box information blob
    label_blob, target_blob, inside_weights_blob, outside_weights_blob = _get_label_blob(roidb)

    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, sublabels_blob)

    blobs = {'data': im_blob,
             'labels': label_blob,
             'targets': target_blob,
             'inside_weights': inside_weights_blob,
             'outside_weights': outside_weights_blob}

    return blobs

def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the different scales.
    """
    num_images = len(roidb)
    processed_ims = []

    for i in xrange(num_images):
        # read image
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        # build image pyramid
        for im_scale in cfg.TRAIN.SCALES_BASE:
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

            processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)

    return blob

def _get_label_blob(roidb):
    """ build the label blob """

    num_images = len(roidb)
    processed_ims_cls = []
    processed_ims_target = []

    for i in xrange(num_images):
        # read image
        im = cv2.imread(roidb[i]['label'], cv2.IMREAD_GRAYSCALE)
        if roidb[i]['flipped']:
            im = im[:, ::-1]

        width = im.shape[1]
        height = im.shape[0]
        im_orig = im.astype(np.int32, copy=True)

        # compute the mask image
        im_mask = im_orig
        im_mask[np.nonzero(im_orig)] = 1

        # compute the class label image
        im_cls = roidb[i]['gt_class'] * im_mask

        # compute the voting label image
        im_target = np.zeros((height, width, 2), dtype=np.float32)
        num_objs = np.amax(im_mask)
        center = np.zeros((2, 1), dtype=np.float32)
        for j in xrange(num_objs):
            y, x = np.where(im_mask == j+1)
            center[0] = (x.max() + x.min()) / 2
            center[1] = (y.max() + y.min()) / 2
            R = np.tile(center, (1, len(x))) - np.vstack((x, y))
            # compute the norm
            N = np.linalg.norm(R, axis=0) + 1e-10
            # normalization
            R = np.divide(R, np.tile(N, (2,1)))
            # assignment
            im_target[y, x, 0] = R[0,:]
            im_target[y, x, 1] = R[1,:]

        # build image pyramid
        for im_scale in cfg.TRAIN.SCALES_BASE:
            im = cv2.resize(im_cls, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_NEAREST)
            processed_ims_cls.append(im)

            im = cv2.resize(im_target, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_NEAREST)
            processed_ims_target.append(im)

    # Create a blob to hold the input images
    blob_cls = im_list_to_blob(processed_ims_cls, 1)
    blob_target = im_list_to_blob(processed_ims_target, 2)

    # blob image size
    image_height = blob_cls.shape[2]
    image_width = blob_cls.shape[3]

    # height and width of the heatmap
    height = np.floor((image_height - 1) / 4.0 + 1)
    height = np.floor((height - 1) / 2.0 + 1 + 0.5)
    height = np.floor((height - 1) / 2.0 + 1 + 0.5)

    width = np.floor((image_width - 1) / 4.0 + 1)
    width = np.floor((width - 1) / 2.0 + 1 + 0.5)
    width = np.floor((width - 1) / 2.0 + 1 + 0.5)

    # rescale the blob
    blob_cls_rescale = np.zeros((num_images, 1, height, width), dtype=np.float32)
    blob_target_rescale = np.zeros((num_images, 2, height, width), dtype=np.float32)
    blob_inside_weights = np.zeros((num_images, 2, height, width), dtype=np.float32)
    blob_outside_weights = np.zeros((num_images, 2, height, width), dtype=np.float32)
    for i in xrange(num_images):
        blob_cls_rescale[i,0,:,:] = cv2.resize(blob_cls[i,0,:,:], dsize=(int(height), int(width)), interpolation=cv2.INTER_NEAREST)
        index = np.where(blob_cls_rescale[i,0,:,:] > 0)
        for j in xrange(blob_target.shape[1]):
            blob_target_rescale[i,j,:,:] = cv2.resize(blob_target[i,j,:,:], dsize=(int(height), int(width)), interpolation=cv2.INTER_NEAREST)
            blob_inside_weights[i,j,index] = 1
            blob_outside_weights[i,j,index] = 1

    return blob_cls_rescale, blob_target_rescale, blob_inside_weights, blob_outside_weights


def _vis_minibatch(im_blob, rois_blob, labels_blob, sublabels_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[2:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        subcls = sublabels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' subclass: ', subcls
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
