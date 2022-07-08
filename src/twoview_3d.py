"""
   This scipt contains functions for two view geometry,
   e.g., estimate essential matrixs, solve relative camera pose... 
   Author: Yan Xu
   Update: Jan 10, 2022
"""

import os, sys
CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(CURRENT_DIR)
import numpy as np
import copy
import cv2


def countCovisiblePoints(pts1, pts2):
    '''
    Count the number of co-visible points from both cameras.
    
    Input:
        pts1: [Nx2], Keypoints of camera 1., "np.nan" if invisible.
        pts2: [Nx2], Keypoints of camera 2.
    Output:
        pts1_covis: Keypoints of camera 1 visible from camera 2.
        pts2_covis: Keypoints of camera 2 visible from camera 1.
        n_covis_pts: Number of co-visible keypoints.
    '''
    visible_1 = ~np.isnan(pts1[:, 0])
    visible_2 = ~np.isnan(pts2[:, 0])
    visible = np.logical_and(visible_1, visible_2)
    visible_idx = np.arange(len(pts1[:, 0]))[visible]
    pts1_covis = pts1[visible]
    pts2_covis = pts2[visible]
    n_covis_pts = len(pts1_covis)
    return pts1_covis, pts2_covis, visible_idx, n_covis_pts


def triangulatePoints(K1, D1, M1, K2, D2, M2, pts1, pts2):
    '''
    Triangulate 3D points, (undistortion before triangulation).
    
    Input:
        K1, D1, M1: intrinsic, extrinsic, distortion parameters of camera 1.
        K2, D2, M2: intrinsic, extrinsic, distortion parameters of camera 2.
        pts1, pts2: [Nx2],point correpondences,np.nan if invisible.
    Output:
        Pts: [Nx3], 3D points w.r.t. cam1.
    '''
    C1, C2 = K1.dot(M1), K2.dot(M2)
    
    pts1_, pts2_, vis_idx, _ = countCovisiblePoints(pts1, pts2)
    pts1_undis = cv2.undistortPoints(pts1_, K1, D1, P=K1).squeeze()
    pts2_undis = cv2.undistortPoints(pts2_, K2, D2, P=K2).squeeze()
    
    Pts_vis = cv2.triangulatePoints(C1, C2, pts1_undis.T, pts2_undis.T).T
    Pts_vis = Pts_vis[:, :3] / np.tile(Pts_vis[:, -1], (3, 1)).T
    Pts = np.nan * np.ones((len(pts1), 3))
    Pts[vis_idx] = Pts_vis
    return Pts


# def estEssentialMatrix(F, K1, K2):
#     '''
#     Compute the essential matrix E, given fundamental matrix F and camera intrinsics K1 and K2.
#         Input:  F, fundamental matrix
#                 K1, internal camera calibration matrix of camera 1
#                 K2, internal camera calibration matrix of camera 2
#         Output: E, the essential matrix
#     '''
#     E = K2.T.dot(F).dot(K1)
#     return E


# def solveMsFromEssenMatOpenCV(E):
#     '''
#     Estimate extrinsic matrix of camera 2 from essential matrix,
#     return all the mathematically possible solutions.
#     '''
#     R1, R2, t = cv2.decomposeEssentialMat(E)
#     M2s = np.zeros([3, 4, 4]).astype('float32')
#     M2s[:,:,0] = np.concatenate([R1, t], axis=1)
#     M2s[:,:,1] = np.concatenate([R1, -t], axis=1)
#     M2s[:,:,2] = np.concatenate([R2, t], axis=1)
#     M2s[:,:,3] = np.concatenate([R2, -t], axis=1)
#     return M2s


# def solveBestM2OpenCV(pts1, pts2, F, K1, D1, K2, D2):

#     # CALCULATE E
#     E = estEssentialMatrix(F, K1, K2)

#     # CALCULATE M1 and M2
#     M1 = np.array([[1., 0., 0., 0.],
#                    [0., 1., 0., 0.],
#                    [0., 0., 1., 0.]])
#     M2_list = solveMsFromEssenMatOpenCV(E)

#     # TRIANGULATION
#     P_best = np.zeros((pts1.shape[0], 3))
#     M2_best = np.zeros((3,4))

#     z_front =  -np.inf
#     for i in range(M2_list.shape[2]):
#         M2 = M2_list[:, :, i]
#         P_i = triangulatePoints(K1, D1, M1, K2, D2, M2, pts1, pts2)
#         z_list = P_i[:, 2]

#         # pick the solution making the most #points in front of the cameras
#         if z_front < (z_list > 0).sum():
#             z_front = (z_list > 0).sum()
#             P_best, M2_best = P_i, M2
#     return P_best, M2_best


def estRelativeCamPose(pts1, pts2, K1, D1, K2, D2):
    '''
    Estimate the relative camera pose from 2D correspondences.
        Input:  pts1, 2D points from the refernce camera.
                pts2, 2D points from the second camera.
                K1, intrinsic matrix of camera 1.
                D1, Distortion vector of camera 1.
                K2, intrinsic matrix of camera 2.
                D2, Distortion vector of camera 2.
        Output: F, the fundamental matrix
                Pts, triangulated 3D points
                M2, the extrinsic matrix of camera 2
    '''
    F, inliers = cv2.findFundamentalMat(pts1, pts2)
    pts1_, pts2_, _, _ = countCovisiblePoints(pts1, pts2)
    _, E, R, t, _ = cv2.recoverPose(pts1_.T, pts2_.T, K1, D1, K2, D2)
    M2 = np.concatenate((R, t), axis=1)
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.]])
    Pts = triangulatePoints(K1, D1, M1, K2, D2, M2, pts1, pts2)
    return F, Pts, M2
