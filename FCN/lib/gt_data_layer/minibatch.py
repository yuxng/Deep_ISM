# --------------------------------------------------------
# Deep ISM
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
    # _vis_minibatch(im_blob, im_depth_blob, label_blob)

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


# backproject pixels into 3D points
def backproject(im_depth, meta_data):

    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # compute projection matrix
    P = meta_data['projection_matrix']
    P = np.matrix(P)
    Pinv = np.linalg.pinv(P)

    # compute the 3D points        
    width = depth.shape[1]
    height = depth.shape[0]
    points = np.zeros((height, width, 3), dtype=np.float32)

    # camera location
    C = meta_data['camera_location']
    C = np.matrix(C).transpose()
    Cmat = np.tile(C, (1, width*height))

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    x3d = Pinv * x2d.transpose()
    x3d[0,:] = x3d[0,:] / x3d[3,:]
    x3d[1,:] = x3d[1,:] / x3d[3,:]
    x3d[2,:] = x3d[2,:] / x3d[3,:]
    x3d = x3d[:3,:]

    # compute the ray
    R = x3d - Cmat

    # compute the norm
    N = np.linalg.norm(R, axis=0)
        
    # normalization
    R = np.divide(R, np.tile(N, (3,1)))

    # compute the 3D points
    X = Cmat + np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

    # compute the azimuth and elevation of each 3D point
    r = np.linalg.norm(X, axis=0)
    # sin of elevation, sin, cos of azimuth
    elevation_sin = np.sin(np.pi/2 - np.arccos(np.divide(X[2,:], r)))
    azimuth_sin = np.sin(np.arctan2(X[1,:], X[0,:]))
    azimuth_cos = np.cos(np.arctan2(X[1,:], X[0,:]))

    points[y, x, 0] = azimuth_sin.reshape(height, width)
    points[y, x, 1] = azimuth_cos.reshape(height, width)
    points[y, x, 2] = elevation_sin.reshape(height, width)

    # mask
    index = np.where(im_depth == 0)
    points[index[0], index[1], :] = 0

    # show the 3D points
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:,:,0], points[:,:,1], points[:,:,2], c='r', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_aspect('equal')
    # plt.show()

    return points


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
