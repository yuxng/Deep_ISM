# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fcn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import cPickle
from utils.blob import im_list_to_blob
import os
import math
import tensorflow as tf

def _get_image_blob(im, im_depth):
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
    im_orig = im_depth.astype(np.float32, copy=True)
    im_orig = im_orig / im_orig.max() * 255
    im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_depth = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_depth.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, blob_depth, np.array(im_scale_factors)


def im_detect(sess, net, im, num_classes):
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
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth)

    # forward pass
    feed_dict={net.data: blobs['data_depth']}
    cls_score, cls_prob = sess.run([net.get_output('cls_score'), net.get_output('cls_prob')], feed_dict=feed_dict)

    scores = cls_prob

    return scores


def vis_detections(im, class_name, dets, thresh=0.1):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    plt.imshow(im)
    for i in xrange(np.minimum(1, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, 4]
        view = dets[i, 6:9]
        if score > thresh:
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3))
            plt.title('{}  {:.3f}, view {:.1f} {:.1f} {:.1f}'.format(class_name, score, \
                      view[0] * 180 / math.pi, view[1] * 180 / math.pi, view[2] * 180 / math.pi))
    plt.show()

def test_net(sess, net, imdb, weights_filename):

    output_dir = get_output_dir(imdb, weights_filename)
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
        scores = im_detect(sess, net, im, imdb.num_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()
        det = {'scores': scores}
        detections[i] = det
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time, _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
