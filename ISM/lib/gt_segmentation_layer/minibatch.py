# --------------------------------------------------------
# Deep ISM
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import sys
import numpy as np
import numpy.random as npr
import cv2
from ism.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import scipy.io
from utils.cython_bbox import bbox_overlaps

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
    label_blob = _get_label_blob(roidb, im_scales, num_classes)

    # For debug visualizations
    # _vis_minibatch(im_blob, im_depth_blob, label_blob)

    blobs = {'data_image': im_blob,
             'data_depth': im_depth_blob,
             'data_label': label_blob}

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
        # rgba
        rgba = cv2.imread(roidb[i]['image'], cv2.IMREAD_UNCHANGED)
        im = rgba[:,:,:3]
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 255

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
        im_depth = im_depth / im_depth.max() * 255
        im_depth = np.tile(im_depth[:,:,np.newaxis], (1,1,3))
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

def _process_label_image(label_image, class_colors):
    """
    change label image to label index
    """
    width = label_image.shape[1]
    height = label_image.shape[0]
    label_index = np.zeros((height, width), dtype=np.float32)

    # label image is in BRG order
    index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
    count = np.zeros((len(class_colors),), dtype=np.float32)
    locations = []
    for i in xrange(len(class_colors)):
        color = class_colors[i]
        ind = 255 * (color[0] + 256*color[1] + 256*256*color[2])
        I = np.where(index == ind)
        locations.append(I)
        count[i] = I[0].shape[0]
        label_index[I] = i
    """
    # classes present in this image
    index_class = np.where(count > 0)[0]
    num_class = len(index_class)

    # sample patches
    num = 32
    pw = 50
    ph = 50
    patches = np.zeros((num, 4), dtype=np.float32)
    for i in range(num):
        # which class to sample from
        j = i % num_class
        cls = index_class[j]
        # sample a location
        while 1:
            ind = np.random.randint(count[cls])
            cx = locations[cls][1][ind]
            cy = locations[cls][0][ind]
            box = np.array([cx-pw/2, cy-ph/2, cx+pw/2, cy+ph/2]).reshape((1,4))
            # compute overlap
            if i > 0:
                overlaps = bbox_overlaps(box.astype(np.float), patches[:i, :].astype(np.float))
                overlap = overlaps.max()
            else:
                overlap = 0
            if box[0,0] > 0 and box[0,1] > 0 and box[0,2] < width and box[0,3] < height and (overlap < 0.9 or count[cls] < pw*ph):
                patches[i,:] = box
                break

    # mask the label
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(num):
        x1 = patches[i, 0]
        y1 = patches[i, 1]
        x2 = patches[i, 2]
        y2 = patches[i, 3]
        mask[y1:y2, x1:x2] = 1

    I = np.where(mask == 0)
    label_index[I] = -1
    """
    
    return label_index


def _get_label_blob(roidb, im_scales, num_classes):
    """ build the label blob """

    num_images = len(roidb)
    processed_ims_cls = []

    for i in xrange(num_images):
        # read label image
        im = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_cls = _process_label_image(im, roidb[i]['class_colors'])

        # rescale image
        im_scale = im_scales[i]
        im = cv2.resize(im_cls, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        processed_ims_cls.append(im)

    # Create a blob to hold the input images
    blob_cls = im_list_to_blob(processed_ims_cls, 1)

    #"""
    # blob image size
    image_height = blob_cls.shape[2]
    image_width = blob_cls.shape[3]

    # height and width of the heatmap
    height = np.floor(image_height / 2.0 + 0.5)
    height = np.floor(height / 2.0 + 0.5)
    height = np.floor(height / 2.0 + 0.5)
    height = np.floor(height / 2.0 + 0.5)
    height = int(height * 8)

    width = np.floor(image_width / 2.0 + 0.5)
    width = np.floor(width / 2.0 + 0.5)
    width = np.floor(width / 2.0 + 0.5)
    width = np.floor(width / 2.0 + 0.5)
    width = int(width * 8)

    # rescale the blob
    blob_cls_rescale = np.zeros((num_images, 1, height, width), dtype=np.float32)
    for i in xrange(num_images):
        blob_cls_rescale[i,0,:,:] = cv2.resize(blob_cls[i,0,:,:], dsize=(width, height), interpolation=cv2.INTER_NEAREST)

    return blob_cls_rescale
    #"""
    # return blob_cls


def _vis_minibatch(im_blob, im_depth_blob, label_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    for i in xrange(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        fig.add_subplot(131)
        plt.imshow(im)

        # show depth image
        im_depth = im_depth_blob[i, :, :, :].transpose((1, 2, 0)).copy()
        im_depth += cfg.PIXEL_MEANS
        im_depth = im_depth[:, :, (2, 1, 0)]
        im_depth = im_depth.astype(np.uint8)
        fig.add_subplot(132)
        plt.imshow(im_depth)

        # show label
        label = label_blob[i, 0, :, :]
        fig.add_subplot(133)
        plt.imshow(label)

        plt.show()
