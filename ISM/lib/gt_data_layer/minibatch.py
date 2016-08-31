# --------------------------------------------------------
# Deep ISM
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
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
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE), size=num_images)
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # build the box information blob
    label_blob, target_blob, inside_weights_blob, outside_weights_blob = _get_label_blob(roidb, im_scales)

    # For debug visualizations
    # _vis_minibatch(im_blob, label_blob, target_blob)

    blobs = {'data': im_blob,
             'labels': label_blob,
             'targets': target_blob,
             'inside_weights': inside_weights_blob,
             'outside_weights': outside_weights_blob}

    return blobs

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_scale = cfg.TRAIN.SCALES_BASE[scale_inds[i]]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)

    return blob, im_scales


def _get_label_blob(roidb, im_scales):
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

        # rescale image
        im_scale = im_scales[i]
        im = cv2.resize(im_cls, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        processed_ims_cls.append(im)
        im = cv2.resize(im_target, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
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
            blob_inside_weights[i,j,index] = 1.0
            blob_outside_weights[i,j,index] = 1.0 / (height * width)

    return blob_cls_rescale, blob_target_rescale, blob_inside_weights, blob_outside_weights


def _vis_minibatch(im_blob, label_blob, target_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(im_blob.shape[0]):
        im = im_blob[i, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        # show image
        plt.subplot(1, 3, 1)
        plt.imshow(im)

        # show label
        label = label_blob[i, 0, :, :]
        print(label.shape, label.max(), label.min())
        plt.subplot(1, 3, 2)
        plt.imshow(label)

        # show the target
        plt.subplot(1, 3, 3)
        plt.imshow(label)
        vx = target_blob[i, 0, :, :]
        vy = target_blob[i, 1, :, :]
        for x in xrange(vx.shape[1]):
            for y in xrange(vx.shape[0]):
                if vx[y, x] != 0 and vy[y, x] != 0:
                    plt.gca().annotate("", xy=(x + vx[y, x], y + vy[y, x]), xycoords='data', xytext=(x, y), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.show()
