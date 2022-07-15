"""
   This scipt contains functions for basic 3D points operations,
   e.g., rotation, fit a plane, 
   Author: Yan Xu
   Update: Jan 07, 2022
"""

import numpy as np
import copy
import cv2


def projectPoints(X, K, D, R, t):
    '''
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    
    Follow the order: [R | t] --> undistortion --> K.
    
    See "http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration
    /camera_calibration.html" or cv2.projectPoints.
    '''
    x = np.asarray(R @ X + np.tile(t, (X.shape[1], 1)).T)
    x[0:2, :] = x[0:2, :] / x[2, :]
    
    r = x[0, :] * x[0,:] + x[1, :] * x[1, :]
    x[0, :] = x[0,:]*(1 + D[0]*r + D[1]*r*r + D[4]*r*r*r) + \
              2*D[2]*x[0,:]*x[1,:] + D[3]*(r + 2*x[0,:]*x[0,:])
    x[1, :] = x[1,:]*(1 + D[0]*r + D[1]*r*r + D[4]*r*r*r) + \
              2*D[3]*x[0,:]*x[1,:] + D[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    
    return x[:2]


def invertExtrinMat(M):
    '''
    Compute the inverse transform of M.
    
    Input:
        M: [3x4], the extrinsic matrix.
    Output:
        M_inv: [3x4], the inverse extrinsic matrix.
    '''
    R, t = M[:, :3], M[:, 3]
    R_ = np.linalg.pinv(R)
    t_ = -R_.dot(t)
    return np.concatenate((R_, np.expand_dims(t_, axis=1)), axis=1)
    # return np.concatenate((R_, t_), axis=1)


def transmitExtrinMats(M32, M21):
    '''
    Compute the relative pose of cam3 w.r.t cam1, M31, given M32 and M21.
    
    Input:
        M21: [3x4], the extrinsic matrix of camera 2 w.r.t. camera 1.
        M32: [3x4], the extrinsic matrix of camera 3 w.r.t. camera 2.
    Output:
        M31: [3x4], the extrinsic matrix of camera 3 w.r.t. camera 1.
    '''
    R21, t21 = M21[:, :3], M21[:, 3]
    R32, t32 = M32[:, :3], M32[:, 3]
    R31 = R32.dot(R21)
    t31 = R32.dot(t21) + t32
    # print(M32.shape, M21.shape, t31, np.squeeze(t31))
    return np.concatenate((R31, np.expand_dims(t31, axis=1)), axis=1)


def affineTransform3D(Pts, M):
    '''
    3D affine transformation.
    
    Input:
        Pts: [Nx3], 3D points w.r.t. camera 1.
        M: [3x4], the extrinsic matrix between camera 2 and 1.
    Output:
        Pts_: [Nx3], 3D points w.r.t. camera 2.
    '''
    return M[:, :3].dot(Pts.T).T + M[:, 3]


def angleBtwnRotatMats(R1, R2):
    '''
    Compute the smallest angle to align two rotation matrixes.
    
    Input:
        R1, R2: [3x3], two rotation matrixes.
    Output:
        theta: the angle (degree).
    '''
    inside_arccos = np.clip((np.trace(R1.dot(R2.T))-1)/2, -1, 1)
    return np.arccos(inside_arccos)*180/np.pi


def solveAbsCamPose(Pts, pts, K, D=None):
    '''
    Solve absolute camera pose using PnP.
    '''
    assert len(Pts) == len(pts)
    vis = np.logical_and(~np.isnan(pts[:, 0]), ~np.isnan(Pts[:, 0]))
    _, rvec, tvec = cv2.solvePnP(Pts[vis], pts[vis], K, D)
    R, _ = cv2.Rodrigues(rvec)
    M = np.concatenate((R, tvec), axis=1)
    return M
