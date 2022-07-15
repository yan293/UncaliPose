"""
   This scipt contains functions for bundle adjustment.
   Author: Yan Xu
   Date: Dec 07, 2021
"""

import os, sys
CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(CURRENT_DIR)

import basic_3d_operations as b3dop
from scipy.sparse import lil_matrix
import scipy.optimize as sciopt
import numpy as np
import json
import time
import cv2
import os


def paramsToVariable(M2s, Pts, fix_cam_pose=False):
    '''
    Convert parameters (M2s and Pts) to vectorized variable x.
    
    Input:
        M2s: [[3x4], ...], a list of M2s (relative extrinsic matrix).
        Pts: [Nx3], 3D points.
    Output:
        x: [1xm], the vectorized variable.
    '''
    if fix_cam_pose:
        return Pts.ravel(order='C')
    
    x = []
    for M2 in M2s:
        r, _ = cv2.Rodrigues(M2[:,:3])
        t = M2[:,3]
        x.append(np.concatenate([r.ravel(order='C'), t.ravel(order='C')]))
    x.append(Pts.ravel(order='C'))
    x = np.concatenate(x)
    return x


def variableToParams(x, M2s, n_Pts, fix_cam_pose=False):
    '''
    Convert parameters (M2s and Pts) to vectorized variable x.
    
    Input:
        x: [1xm], the vectorized variable.
        M2s: [[3x4], ...], a list of M2s, when fixed camera pose,
            these are GT M2s.
        n_Pts: number of 3D points.
    Output:
        M2s: [[3x4], ...], a list of M2s (relative extrinsic matrix).
        Pts: [Nx3], 3D points.
    '''
    if fix_cam_pose:
        return M2s, x.reshape((-1, 3))
    
    Pts = x[-n_Pts*3:].reshape((-1, 3))
    rts = x[:-n_Pts*3].reshape((-1, 6))
    M2s = []
    for rt in rts:
        r, t = rt[:3], rt[3:6]
        R, _ = cv2.Rodrigues(r)
        M2s.append(np.concatenate([R, t.reshape((len(t), 1))], axis=1))
    return M2s, Pts


def visiblePtsIndex(pts):
    assert pts.shape[1] == 2 or pts.shape[1] == 3
    return np.arange(len(pts))[~np.isnan(pts[:, 0])]


def rodriguesResidual(
        K1, D1, M1, p1, K2s, D2s, M2s, p2s, x, fix_cam_pose=False):
    '''
    Rodrigues residual.
    
    Input:
        K1, [3x3], the intrinsic matrix of camera 1 (reference camera).
        D1, [1x5], the distortion parameters of camera 1.
        M1, [3x4], the extrinsic matrix of camera 1.
        p1, [Nx2], 2D points observed from camera 1, (np.nan if invisible).
        K2s, [[3x3], ...], a list of intrinsic matrixs of camera 2s.
        D2s, [[1x5], ...], a list of distortion parameters of camera 2s.
        p2s, [[Nx2], ...], a list of extrinsic matrix of camera 2s.
        x, the flattened concatenation of r2s, t2s, and P (3D points).
    Output:
        residuals, 2D reprojection residual.
    '''
    n_Pts = len(p1)
    M2s, P = variableToParams(x, M2s, n_Pts, fix_cam_pose=fix_cam_pose)
    
    vis1 = visiblePtsIndex(p1)
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    p1_ = b3dop.projectPoints(P[vis1].T, K1, D1, M1[:,:3], M1[:,3])
    reproj_err = [p1_.T - p1[vis1]]
    
    for i in range(len(K2s)):
        R2, t2 = M2s[i][:,:3], M2s[i][:,3]
        vis2 = visiblePtsIndex(p2s[i])
        p2_ = b3dop.projectPoints(P[vis2].T, K2s[i], D2s[i], R2, t2)
        reproj_err.append(p2_.T - p2s[i][vis2])
    
    residuals = np.concatenate(reproj_err, axis=0).ravel(order='C')
    return residuals


def jacobianSparsity(pts_list, fix_cam_pose=False):
    '''
    Construct a sparse Jocobian matrix.
    
    Input:
        pts_list:[[Nx2], [Nx2], ...], 2D points (from all viws) list,
            np.nan if a point is invisible from a camera view.
     Output:
        A: Jacobian sparcity matrix.
    '''
    n_Pts, n_cam = len(pts_list[0]), len(pts_list)
    n_cam2 = n_cam - 1  # exclude the world camera
    
    n_var = n_cam2 * 6 + n_Pts * 3  # number of variables
    visible_list = [visiblePtsIndex(x) for x in pts_list]
    n_res = np.sum([len(x) for x in visible_list]) * 2  # number of residuals
    
    A = lil_matrix((n_res, n_var), dtype=int)
    
    # points observed from the world camera are at the beginning
    visPts_cam1 = np.array(visible_list[0])
    for i in range(3):
        A[:len(visPts_cam1)*2, n_cam2*6+visPts_cam1*3+i] = 1
        
    # points observed from other cameras
    start = len(visPts_cam1) * 2
    for k in range(1, len(visible_list)):
        visPts_cam2 = np.array(visible_list[k])
        end = start + len(visPts_cam2) * 2
        A[start:end, (k-1)*6:k*6] = 1  # camera paramters derivative
        for i in range(3): 
            A[start:end, n_cam2*6+visPts_cam2*3+i] = 1  # 3D points derivative
        start = end
        
    # see if optimize camera pose together with 3D points
    if fix_cam_pose:
        A = A[:, n_cam2*6:]
    return A


def bundleAdjustment(K1, D1, M1, p1, K2s, D2s, M2s, p2s, Pts, Pts_prev=None,
                     fix_cam_pose=False, verbose=2, max_iter=100, alpha=1,
                     save_dir=None, wrld_cam_id=None):
    '''
    Bundle adjustment.
    
    Input:
        K1, [3x3], the intrinsic matrix of camera 1 (reference camera).
        D1, [1x5], the distortion parameters of camera 1.
        M1, [3x4], the extrinsic matrix of camera 1.
        p1, [N1x2], 2D points observed from camera 1.
        K2s, [[3x3], ...], a list of intrinsic matrixs of camera 2s.
        D2s, [[1x5], ...], a list of distortion parameters of camera 2s.
        M2s, [[3x4], ...], a list of extrinsic matrixs of camera 2s.
        p2s, [[N2x2], ...], a list of extrinsic matrix of camera 2s.
        Pts, [Nx3], 3D points. 
    Output:
        M2s_BA, [[3x4], ...], a list of extrinsic matrixs of camera 2s.
        Pts_BA, [Nx3], 3D points after Bundle Adjustment.
    '''
    if verbose:
        print("\nBundle adjustment:\n====================")
    
    # --- pre-step: check in case there are invisible 3D Points
    vis_ids = np.arange(len(Pts))[~np.isnan(Pts[:, 0])]
    Pts_ = Pts[vis_ids]
    p1_ = p1[vis_ids]
    p2s_ = [p2[vis_ids] for p2 in p2s]

    func = lambda x: rodriguesResidual(
        K1, D1, M1, p1_, K2s, D2s, M2s, p2s_, x,fix_cam_pose=fix_cam_pose)
    jac_spars = jacobianSparsity([p1_]+list(p2s_),fix_cam_pose=fix_cam_pose)
    
    x_init = paramsToVariable(M2s, Pts_, fix_cam_pose=fix_cam_pose)
    res = sciopt.least_squares(
        func, x_init, verbose=verbose, jac_sparsity=jac_spars,
        max_nfev=max_iter, loss='linear')
    
    M2s_BA, Pts_BA_vis = variableToParams(
        res.x, M2s, len(Pts_), fix_cam_pose=fix_cam_pose)
    
    # --- post-step: in case there are invisible 3D Points
    Pts_BA = np.nan * np.ones(Pts.shape)
    Pts_BA[vis_ids] = Pts_BA_vis
    
    if save_dir is not None:
        assert wrld_cam_id is not None
        Ms = M2s_BA.copy()
        Ms.insert(wrld_cam_id, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        save_data = {
            'Pts': Pts_BA.tolist(),
            'Ms': np.array(Ms).tolist(),
            'world_camera_id': wrld_cam_id}
        json.dump(save_data, open(os.path.join(save_dir,'pose_est.json'),'w'))
    if verbose:
        print("\nDone.\n====================\n")
    return M2s_BA, Pts_BA


def bundleAdjustmentWrapper(BA_input, max_iter=100, fix_cam_pose=False,
                            Pts_prev=None, verbose=True, alpha=1,
                            wrld_cam_id=None, save_dir=None):
    '''
    Bundle adjustment with input wrapped as a dictionary.
    '''
    K1, D1, M1 = BA_input['K1'], BA_input['D1'], BA_input['M1']
    K2s, D2s, M2s = BA_input['K2s'], BA_input['D2s'], BA_input['M2s']
    p1, p2s, P = BA_input['p1'], BA_input['p2s'], BA_input['P']
    
    verbose_num = 2 if verbose else 0
    M2s_BA, Pts_BA = bundleAdjustment(
        K1, D1, M1, p1, K2s, D2s, M2s, p2s, P, Pts_prev=Pts_prev,
        fix_cam_pose=fix_cam_pose, max_iter=max_iter, alpha=alpha,
        verbose=verbose_num, save_dir=save_dir, wrld_cam_id=wrld_cam_id
    )
    return Pts_BA, M2s_BA
