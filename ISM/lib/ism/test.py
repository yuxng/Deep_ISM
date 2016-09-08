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
from utils.blob import im_list_to_blob
import os
import math
import scipy.io
from scipy.optimize import minimize

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


# backproject pixels into 3D points
def backproject_camera(im_depth, meta_data):

    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # get intrinsic matrix
    K = meta_data['intrinsic_matrix']
    K = np.matrix(K)
    Kinv = np.linalg.inv(K)

    # compute the 3D points        
    width = depth.shape[1]
    height = depth.shape[0]
    points = np.zeros((height, width, 3), dtype=np.float32)

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = Kinv * x2d.transpose()

    # compute the norm
    N = np.linalg.norm(R, axis=0)
        
    # normalization
    R = np.divide(R, np.tile(N, (3,1)))

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
    points[y, x, 0] = X[0,:].reshape(height, width)
    points[y, x, 1] = X[1,:].reshape(height, width)
    points[y, x, 2] = X[2,:].reshape(height, width)

    # mask
    index = np.where(im_depth == 0)
    points[index[0], index[1], :] = 0

    return points


def loss_pose(x, points, im_depth, center_pred):
    """ loss function for pose esimation """
    rx = x[0]
    ry = x[1]
    rz = x[2]
    C = x[3:6].reshape((3,1))

    # construct rotation matrix
    Rx = np.matrix([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.matrix([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.matrix([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz * Ry * Rx

    # transform the points
    index = np.where(im_depth > 0)
    x3d = points[index[0], index[1], :].transpose()
    num = x3d.shape[1]
    Cmat = np.tile(C, (1, num))
    X = R * (x3d - Cmat)

    # compute the azimuth and elevation of each 3D point
    r = np.linalg.norm(X, axis=0)
    elevation = (np.pi/2 - np.arccos(np.divide(X[2,:], r))) / np.pi
    azimuth = np.arctan2(X[1,:], X[0,:]) / np.pi

    # get the predicted azimuth and elevation
    azimuth_pred = center_pred[0, 2, index[0], index[1]]
    elevation_pred = center_pred[0, 3, index[0], index[1]]

    # compute the loss
    loss = (np.mean(np.power(azimuth - azimuth_pred, 2)) + np.mean(np.power(elevation - elevation_pred, 2))) / 2
    return loss


def pose_estimate(im_depth, meta_data, center_pred):
    """ estimate the pose of object from network predication """
    # compute 3D points in camera coordinate framework
    points = backproject_camera(im_depth, meta_data)

    # rescale the 3D point map
    height = center_pred.shape[2]
    width = center_pred.shape[3]
    im_depth_rescale = cv2.resize(im_depth, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
    points_rescale = cv2.resize(points, dsize=(height, width), interpolation=cv2.INTER_NEAREST)

    # optimization
    # initialization
    x0 = np.zeros((6,1), dtype=np.float32)
    index = np.where(im_depth > 0)
    x3d = points[index[0], index[1], :]
    x0[3:6] = np.mean(x3d, axis=0).reshape((3,1))
    bounds = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (None, None), (None, None), (np.min(x3d[:,2]), None))
    res = minimize(loss_pose, x0, (points_rescale, im_depth_rescale, center_pred), method='SLSQP', bounds=bounds, options={'disp': True})
    print res.x

    # transform the points
    rx = res.x[0]
    ry = res.x[1]
    rz = res.x[2]
    C = res.x[3:6].reshape((3,1))

    # construct rotation matrix
    Rx = np.matrix([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.matrix([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.matrix([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz * Ry * Rx

    # transform the points
    index = np.where(im_depth_rescale > 0)
    x3d = points_rescale[index[0], index[1], :].transpose()
    num = x3d.shape[1]
    Cmat = np.tile(C, (1, num))
    points_transform = R * (x3d - Cmat)

    return points_rescale, np.array(points_transform)


def vis_detections(im, cls_prob, center_pred, points_rescale, points_transform):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    im = im[:, :, (2, 1, 0)]
    fig = plt.figure()

    # show image
    fig.add_subplot(241)
    plt.imshow(im)

    # show class label
    label = cls_prob[0, 1, :, :]
    fig.add_subplot(242)
    plt.imshow(label)

    # show the target
    # ax = fig.add_subplot(133, projection='3d')
    # ax.scatter(center_pred[0, 0, :, :], center_pred[0, 2, :, :], center_pred[0, 1, :, :], c='r', marker='o')
    fig.add_subplot(243)
    plt.imshow(label)
    vx = center_pred[0, 0, :, :]
    vy = center_pred[0, 1, :, :]
    for x in xrange(vx.shape[1]):
        for y in xrange(vx.shape[0]):
            if vx[y, x] != 0 and vy[y, x] != 0:
                plt.gca().annotate("", xy=(x + vx[y, x], y + vy[y, x]), xycoords='data', xytext=(x, y), textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # show the azimuth image
    azimuth = center_pred[0, 2, :, :]
    fig.add_subplot(245)
    plt.imshow(azimuth)

    # show the elevation image
    elevation = center_pred[0, 3, :, :]
    fig.add_subplot(246)
    plt.imshow(elevation)

    # show the 3D points
    ax = fig.add_subplot(247, projection='3d')
    ax.scatter(points_rescale[:,:,0], points_rescale[:,:,1], points_rescale[:,:,2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    # show the 3D points transform
    ax = fig.add_subplot(248, projection='3d')
    ax.scatter(points_transform[0,:], points_transform[1,:], points_transform[2,:], c='r', marker='o')
    ax.scatter(0, 0, 0, c='g', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

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

    perm = np.random.permutation(np.arange(num_images))

    # for i in xrange(num_images):
    for i in perm:
        im = cv2.imread(imdb.image_path_at(i))

        # shift
        # rows = im.shape[0]
        # cols = im.shape[1]
        # M = np.float32([[1,0,50],[0,1,25]])
        # im = cv2.warpAffine(im,M,(cols,rows))

        # rescaling
        # im = cv2.resize(im, None, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)

        _t['im_detect'].tic()
        cls_prob, center_pred = im_detect(net, im, imdb.num_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()
        det = {'cls_prob': cls_prob, 'center_pred': center_pred}
        detections[i] = det
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time, _t['misc'].average_time)

        # read depth image
        im_depth = cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED)
        # read meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))
        # compute object pose
        points_rescale, points_transform = pose_estimate(im_depth, meta_data, center_pred)

        vis_detections(im, cls_prob, center_pred, points_rescale, points_transform)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
