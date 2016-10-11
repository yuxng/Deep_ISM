# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
import normals.gpu_normals
from fcn.config import cfg
import cv2

# backproject pixels into 3D points
def backproject(im_depth, meta_data):

    # convert depth
    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']
    near = meta_data['near_plane']
    far = meta_data['far_plane']
    depth = (far + near) / (far - near) - (2 * far * near) / ((far - near) * depth)
    depth = (depth + 1) / 2

    # compute projection matrix
    P = meta_data['projection_matrix']
    P = np.matrix(P)
    Pinv = np.linalg.pinv(P)

    # construct the 2D points matrix
    width = depth.shape[1]
    height = depth.shape[0]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, height-1-y, depth, ones), axis=2).reshape(width*height, 4)

    # Map x and y from window coordinates
    viewport = meta_data['viewport']
    x2d[:, 0] = (x2d[:, 0] - viewport[0]) / viewport[2];
    x2d[:, 1] = (x2d[:, 1] - viewport[1]) / viewport[3];

    # Map to range -1 to 1
    x2d[:, 0] = x2d[:, 0] * 2 - 1;
    x2d[:, 1] = x2d[:, 1] * 2 - 1;
    x2d[:, 2] = x2d[:, 2] * 2 - 1;

    # backprojection
    x3d = Pinv * x2d.transpose()
    x3d[0,:] = x3d[0,:] / x3d[3,:]
    x3d[1,:] = x3d[1,:] / x3d[3,:]
    x3d[2,:] = x3d[2,:] / x3d[3,:]
    x3d = x3d[:3,:].astype(np.float32)

    norms = normals.gpu_normals.gpu_normals(x3d, width, height, cfg.GPU_ID)

    # convert normals to an image
    N = np.zeros((height, width, 3), dtype=np.float32)
    N[y, x, 0] = norms[:, 0].reshape(height, width)
    N[y, x, 1] = norms[:, 1].reshape(height, width)
    N[y, x, 2] = norms[:, 2].reshape(height, width)
    N = 127.5*N + 127.5
    N = N.astype(np.uint8)    

    # show the 3D points
    if 0:
        # construct the 3D points        
        points = np.zeros((height, width, 3), dtype=np.float32)
        points[y, x, 0] = x3d[0, :].reshape(height, width)
        points[y, x, 1] = x3d[1, :].reshape(height, width)
        points[y, x, 2] = x3d[2, :].reshape(height, width)

        ns = np.zeros((height, width, 3), dtype=np.float32)
        ns[y, x, 0] = norms[:, 0].reshape(height, width)
        ns[y, x, 1] = norms[:, 1].reshape(height, width)
        ns[y, x, 2] = norms[:, 2].reshape(height, width)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(131, projection='3d')
        perm = np.random.permutation(np.arange(height*width))
        index = perm[:10000]
        X = points[:,:,0].flatten()
        Y = points[:,:,2].flatten()
        Z = points[:,:,1].flatten()
        # U = ns[:,:,0].flatten()
        # V = ns[:,:,2].flatten()
        # W = ns[:,:,1].flatten()
        ax.scatter(X[index], Y[index], Z[index], c='r', marker='o')
        # ax.quiver(X[index], Y[index], Z[index], U[index], V[index], W[index], length=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')

        fig.add_subplot(132)
        plt.imshow(im_depth)

        fig.add_subplot(133)
        plt.imshow(N)
        plt.show()

    return N
