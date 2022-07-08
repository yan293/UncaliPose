"""
   This scipt contains functions for multi-view 3D reconstruction,
   e.g., 
   Author: Yan Xu
   Update: Dec 07, 2021
"""

import os, sys
CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(CURRENT_DIR)
import basic_3d_operations as b3dop
import bundle_adjustment as buad
import twoview_3d as twov3d
import pickle as pkl
import numpy as np
import json
import cv2


def solveRelPoseWrtCam(M2_dict, cam1):
    '''
    Compute the relative pose w.r.t. a chosen camera.
    
    Input:
        M2_dict: A dictionary of M2s, {'M_1_0': [3x4], ...}, 'M_1_0'
                 means the pose of camera 1 w.r.t. camera 0.
        cam1: Index of the chosen camera, from 0 to N-1.
    Output:
        M2s_wrt_cam1: A dictionary of M2s w.r.t the chosen camera 1.
                      An example: {'M_1_0': {
                      'direct': [3x4], 'indirect':[[3x4],[3x4],...]
                      }, {...}, {...}, ...}. 'indirect' means the M2
                      computed using an intermediate camera 3.
    '''
    cam_set = set()
    for M2_name in M2_dict.keys():
        cam_set.add(int(M2_name.split('_')[-1]))
        cam_set.add(int(M2_name.split('_')[-2]))
    cam2_list = list(cam_set)
    cam2_list.remove(cam1)
    
    M2s_wrt_cam1 = {}
    for cam2 in cam2_list:
        if 'M_' + str(cam2) + '_' + str(cam1) in M2_dict:
            M2_direct = M2_dict['M_' + str(cam2) + '_' + str(cam1)]
        else:
            M2_direct = None
            
        cam3_list = cam2_list.copy()
        cam3_list.remove(cam2)
        M2s_indirect = []
        for cam3 in cam3_list:
            if 'M_' + str(cam2) + '_' + str(cam3) in M2_dict and \
               'M_' + str(cam3) + '_' + str(cam1) in M2_dict:  
                M23 = M2_dict['M_' + str(cam2) + '_' + str(cam3)]
                M31 = M2_dict['M_' + str(cam3) + '_' + str(cam1)]
                M21_ = b3dop.transmitExtrinMats(M23, M31)
                M2s_indirect.append(M21_)

        M2s_wrt_cam1['M_'+str(cam2)+'_'+str(cam1)] = {
            'direct': M2_direct,
            'indirect': M2s_indirect}
    return M2s_wrt_cam1


def findBestCovisCamPairs(pts_corresp):
    '''
    For each camera, find the camera with most covisible keypoints.
    
    Input:
        pts_corresp: Point correspondence list [[Nx2], [Nx2], ....],
                     invisible key points are represented as np.nan.
    Output:
        most_covis_pts_cam: A dictionary, where the key is a camera
            (idx) and the value is (best_cam2, # covis_pts).
    '''
    most_covis_pts_cam = {}
    n_cams = len(pts_corresp)
    
    for cam1 in range(n_cams - 1):
        pts1 = pts_corresp[cam1]
        covis_pts_count = 0
        best_cam = -1
        
        for cam2 in range(cam1 + 1, n_cams):
            pts2 = pts_corresp[cam2]
            _, _, _, n_covis_pts = twov3d.countCovisiblePoints(pts1, pts2)
            
            if cam2 in most_covis_pts_cam and \
               most_covis_pts_cam[cam2][1] >= n_covis_pts:
                pass
            else:
                most_covis_pts_cam[cam2] = (cam1, n_covis_pts)
            
            if n_covis_pts <= covis_pts_count:
                continue
            covis_pts_count = n_covis_pts
            best_cam = cam2
            
        # --- see if the camera is in the dict
        if cam1 in most_covis_pts_cam and \
           most_covis_pts_cam[cam1][1] >= covis_pts_count:
            pass
        else:
            most_covis_pts_cam[cam1] = (best_cam, covis_pts_count)
    return most_covis_pts_cam


def selfValidateM2(M2_list):
    '''
    Find the best M2 that is closest (on average) to all other M2s.
    We assume that most M2s are close to the correct value.
    
    Input:
        M2_list: [[3x4],...], a list of M2 (extrinsic) matrix.
    Output:
        best_M2, the best M2.
    '''
    M2_idx_metric_pair = []
    for i in range(len(M2_list)):
        angles_to_other_M2s = []
        for j in range(len(M2_list)):
            if j == i: continue
            angles_to_other_M2s.append(
                b3dop.angleBtwnRotatMats(M2_list[i][:,:3], M2_list[j][:,:3]))
        M2_idx_metric_pair.append((i, np.mean(angles_to_other_M2s)))
        
    M2_idx_metric_pair = np.array(M2_idx_metric_pair)
    best_M2 = M2_list[int(
        M2_idx_metric_pair[M2_idx_metric_pair[:,1].argsort()[0]][0])]
    return best_M2


def findValidM2sForAllCams(M2_dict, M2_angle_thold=15):
    '''
    Find valid M2s for cameras w.r.t. all cameras.
    
    Input:
        M2_dict: A dictionary of M2s, {'M_1_0':[3x4], ...},
                 'M_1_0' means the pose of cam1 w.r.t. cam0.
        M2_angle_thold: the angle sensitivity that dicides whether
            two matrixes are rotation-wise close.
    Output:
        valid_M2s_dict, valid M2s.
    '''
    # count number of cameras
    cam_ids = set()
    for key in M2_dict.keys():
        cam1_id = int(key.split('_')[1])
        cam2_id = int(key.split('_')[2])
        cam_ids.add(cam1_id)
        cam_ids.add(cam2_id)
    
    # select valid M2s for each camera
    valid_M2s_dict = {}
    for wrld_cam_id in cam_ids:
        M2s_wrt_cam1 = solveRelPoseWrtCam(M2_dict, wrld_cam_id)
        for key, val in M2s_wrt_cam1.items():
            M2_list = list(val['indirect'])
            if val['direct'] is not None:
                M2_list = [val['direct']] + list(val['indirect'])
            best_M2 = selfValidateM2(M2_list)
            
            # give "direct" estimation more trust
            if val['direct'] is not None:
                angle=b3dop.angleBtwnRotatMats(
                    val['direct'][:,:3], best_M2[:,:3])
                if angle < M2_angle_thold:
                    best_M2 = val['direct']
                
            valid_M2s_dict[key] = best_M2
    return valid_M2s_dict


def solveMultiCamRelPose(pts_corresp, cams_list, wrld_cam_id=None):
    '''
    Estimate camera pose w.r.t a selected camera for multi-cameras.
    
    Input:
        pts_corresp: Point correspondence list [[Nx2], [Nx2], ....],
                     invisible key points are represented as np.nan.
        cams_list: A list of cameras, each element of the list is a
                   dictionary, containing necessary intrin/extrin
                   information, an example of panoptic data would be,
                   [{'name': ..., 'resolution': ..., 'K': [3x3],
                   'distCoef': [1x4/1x5], 'R': [3x3], 't': [3x1]}, ...].
        wrld_cam_id: world camera id, if None, all cameras.
    Output:
        M2_dict: A dictionary of M2s: {'M_1_0':[3x4],...}, 'M_1_0'
                 means the pose of camera 1 w.r.t. camera 0.
    '''
    if wrld_cam_id is not None:
        wrld_cam_list = [wrld_cam_id]
    else:
        wrld_cam_list = list(np.arange(len(cams_list)))
        
    M2_dict = {}
    for cam1_id in wrld_cam_list:
        cam1 = cams_list[cam1_id]
        K1, D1 = cam1['K'], cam1['distCoef']
        pts1 = pts_corresp[cam1_id]
        for cam2_id in range(len(cams_list)):
            if cam2_id == cam1_id:
                continue
            cam2 = cams_list[cam2_id]
            K2, D2 = cam2['K'], cam2['distCoef']
            pts2 = pts_corresp[cam2_id]
            pts1_, pts2_, _, _ = twov3d.countCovisiblePoints(pts1, pts2)
            if len(pts1_) != 0:
                _, _, M2 = twov3d.estRelativeCamPose(pts1_,pts2_,K1,D1,K2,D2)
                M2_dict['M_' + str(cam2_id) + '_' + str(cam1_id)] = M2
    return M2_dict


def triangulateWrtAllCams(
        pts_corresp_list, cam_param_list, n_person, M2_angle_thold=15,
        verbose=True):
    '''
    Triangulate 3D points w.r.t all cameras.
    
    Input:
        pts_corresp_list: 2D point correspondences list.
        cam_param_list: A list of camera parameters (intrin, extrin, ...).
        M2_angle_thold: The sensitivity of angle between M2 matrixes.
    Output:
        Pts_list: [[Nx3], ...], A list of 3D points w.r.t. all cameras.
        visible_list: [[[1 x n_joints], [1 x n_joints], ...], ...], info
            about which joint of which person is visible from which camera.
    '''
    if verbose:
        print('Triangulate w.r.t. all the cameras.')
    n_cam = len(pts_corresp_list)
    n_joints = len(pts_corresp_list[0]) // n_person
        
    M2s_wrt_all_cams = solveMultiCamRelPose(pts_corresp_list, cam_param_list)
    valid_M2s = findValidM2sForAllCams(
        M2s_wrt_all_cams, M2_angle_thold=M2_angle_thold)
    
    Pts_list, visible_list =[], []
    for cam1_id in range(n_cam):
        cam2_id = findBestCovisCamPairs(pts_corresp_list)[cam1_id][0]
        cam1, cam2 = cam_param_list[cam1_id], cam_param_list[cam2_id]
        K1, K2 = cam1['K'], cam2['K']
        D1, D2 = cam1['distCoef'], cam2['distCoef']
        M1 = np.concatenate([np.eye(3), np.zeros((3,1))], axis=1)
        M2 = valid_M2s['M_'+str(cam2_id)+'_'+str(cam1_id)]
        pts1, pts2 = pts_corresp_list[cam1_id], pts_corresp_list[cam2_id]
        Pts_wrt_cam1 = twov3d.triangulatePoints(K1,D1,M1,K2,D2,M2,pts1,pts2)
        Pts_list.append(Pts_wrt_cam1)
        
        visible = []
        for i in range(n_person):
            Person = Pts_wrt_cam1[i*n_joints:(i+1)*n_joints]
            visible.append(np.where(~np.isnan(Person[:, 0]))[0])
        visible_list.append(visible)
    return Pts_list, visible_list, valid_M2s


def selectRefCam(pts_corresp):
    '''
    Select the reference camera.  This camera should be the camera
    that has the most co-visible points with other cameras.
    
    Input:
        pts_corresp: Point correspondence list [[Nx2], ...],
            invisible key points are represented as np.nan.
    Output:
        wrld_cam_id: world camera id, integer, e.g. 0, 1,...
    '''
    most_covis_pts_cam = findBestCovisCamPairs(pts_corresp)
    cam_count = {}
    for key, val in most_covis_pts_cam.items():
        if val[0] not in cam_count:
            cam_count[val[0]] = 1
        else:
            cam_count[val[0]] += 1
    
    max_count = -1
    for cam_id, count in cam_count.items():
        if max_count < count:
            wrld_cam_id = cam_id
            max_count = count
    return wrld_cam_id


def merge3DSkeletonsWrtTwoCams(Pts1, Pts2):
    '''
    Merge the Pts2 to the coordinate system of Pts1.
    
    Input:
        Pts1, Pts2: [Nx3], 3D coordinates under two camera systems.
    Output:
        Pts: [Nx3], merged 3D coordinates under camera 1.
    '''
    if np.sum(~np.isnan(Pts1[:, 0])) < 2:
        print("This person is invisible from the world camera.")
        return Pts1
    vis_id1 = np.where(~np.isnan(Pts1[:, 0]))[0]
    vis_id2 = np.where(~np.isnan(Pts2[:, 0]))[0]
        
    # step 1: find a visible point pair from both cameras
    for i in range(len(Pts1) - 1):
        if ~np.isnan(Pts1[i, 0]) and ~np.isnan(Pts2[i, 0]): break
    for j in range(i + 1, len(Pts1) - 1):
        if ~np.isnan(Pts1[j, 0]) and ~np.isnan(Pts2[j, 0]): break
    if not ('i' in locals() and 'j' in locals()):
        return Pts1
    
    P1_cam1, P2_cam1 = Pts1[i], Pts1[j]
    P1_cam2, P2_cam2 = Pts2[i], Pts2[j]
    ratio = np.linalg.norm(P2_cam1 - P1_cam1, ord=2) / \
            np.linalg.norm(P2_cam2 - P1_cam2, ord=2)
    Pts2_rela = Pts2 - P1_cam2
    
    # step 2: merge Pts1 and Pts2
    Pts = P1_cam1 + Pts2_rela * ratio
    Pts[vis_id1] = Pts1[vis_id1]
    return Pts


def merge3DJointsFromMultiViews(Pts_list, M2s_dict, n_joints, wrld_cam_id=0,
                                verbose=True):
    '''
    Merge all the 3D points in "Pts_list".
    
    Input:
        Pts_list: [[Nx3], ...], 3D points at different coordinate systems.
        M2s_dict: A dictionary of M2s, {'M_1_0':[3x4], ...}, 'M_1_0' means
            the pose of cam1 w.r.t. cam0.
        n_joints: int, number of joints per person.
        wrld_cam_id: world camera view to merge to.
    Output:
        Pts1: [Nx3], merged 3D points at coordinate system "world_cam".
    '''
    if verbose:
        print('Use all cameras as the world camera and merge information.')
    n_cam = len(Pts_list)
    n_person = len(Pts_list[0]) // n_joints
    Pts1 = Pts_list[wrld_cam_id]
    
    # step 0: convert points to the same coordinate system
    Pts2_list = []
    for i in range(n_cam):
        if i == wrld_cam_id: continue
        Pts2_list.append(b3dop.affineTransform3D(
            Pts_list[i], M2s_dict['M_' + str(wrld_cam_id) + '_' + str(i)]))
    
    # step 1: merge visible people one-by-one
    for Pts2 in Pts2_list:
        persons = []
        for i in range(n_person):
            P1 = Pts1[i * n_joints:(i + 1) * n_joints]
            P2 = Pts2[i * n_joints:(i + 1) * n_joints]
            persons.append(merge3DSkeletonsWrtTwoCams(P1, P2))
        Pts1 = np.concatenate(persons, axis=0)

    # step 2: merge invisible people all at once
    for Pts2 in Pts2_list:
        Pts1 = merge3DSkeletonsWrtTwoCams(Pts1, Pts2)
        
    visible = []
    for i in range(n_person):
        Person = Pts1[i * n_joints:(i + 1) * n_joints]
        visible.append(np.where(~np.isnan(Person[:, 0]))[0])
        
    return Pts1, visible


def solve3DPose(
        pts_corresp_dict, cam_params_dict, n_person, wrld_cam_id=None,
        M2_angle_thold=15, Pts_prev=None, verbose=True):
    '''
    Given point correspondences and camera parameters of a multi-camera
    system, solve 3D human poses and relative camera poses.
    
    Input:
        pts_corresp_dict: 2D-2D point correspondences, {cam_id: {
            'keypoints': [Nx2], 'box_files': []}, ...}.
        cam_params_dict: camera parameters, {cam_id: {
            'K': [3x3], 'distCoef': [4x1/5x1]}, ...}
        n_person: number of person.
        wrld_cam_id: world camera id, if "None", will be auto-selected.
    Output:
        Pts: solved 3D points coordinates [Nx3].
        BA_input: prepared data for bundle adjustment.
    '''
    if verbose:
        print("\nSolve 3D human and camera poses:\n====================")
        
    pts_corresp_list, cam_params_list, cam_names = [], [], []
    for cam_id in sorted(list(pts_corresp_dict.keys())):
        pts_corresp_list.append(pts_corresp_dict[cam_id]['keypoints'])
        cam_params_list.append(cam_params_dict[cam_id])
        cam_names.append(cam_id)
    n_joints = len(pts_corresp_list[0]) // n_person

    Pts_list, visible_list, M2s_dict = triangulateWrtAllCams(
        pts_corresp_list, cam_params_list, n_person,
        M2_angle_thold=M2_angle_thold, verbose=verbose)
    
    if wrld_cam_id is None:
        wrld_cam_id = selectRefCam(pts_corresp_list)
        if verbose:
            print("Auto-select cam {} as world camera.".format(wrld_cam_id))
    else:
        if verbose:
            print("Manual-select cam {} as world camera.".format(wrld_cam_id))
    wrld_cam_name = cam_names[wrld_cam_id]
    
    Pts, visible = merge3DJointsFromMultiViews(
        Pts_list, M2s_dict, n_joints, wrld_cam_id=wrld_cam_id,verbose=verbose)
    
    if Pts_prev is not None:
        Pts = merge3DSkeletonsWrtTwoCams(Pts, Pts_prev)
    
    # prepare data for bundle adjustment
    K2s, D2s, M2s, p2s = [], [], [], []
    for i, cam_params in enumerate(cam_params_list):
        if i == wrld_cam_id:
            p1 = pts_corresp_list[i]
            K1 = cam_params['K']
            D1 = cam_params['distCoef']
            M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            continue
        p2s.append(pts_corresp_list[i])
        K2s.append(cam_params['K'])
        D2s.append(cam_params['distCoef'])
        
        _, _, _, n_covis = twov3d.countCovisiblePoints(
            Pts[:, :2], pts_corresp_list[i])
        if n_covis >= 10:   
            M2 = b3dop.solveAbsCamPose(
                Pts, pts_corresp_list[i],cam_params['K'],D=cam_params['distCoef'])
        else:
            M2 = M2s_dict['M_' + str(i) + '_' + str(wrld_cam_id)]
        # M2 = M2s_dict['M_' + str(i) + '_' + str(wrld_cam_id)]  # works better
        M2s.append(M2)
    BA_input = {
        'p1': p1, 'p2s': p2s, 'P': Pts,
        'K1': K1, 'D1': D1, 'M1': M1,
        'K2s': K2s, 'D2s': D2s, 'M2s': M2s}
    if verbose:
        print("====================\n")
    return Pts_list, Pts, BA_input, wrld_cam_name, wrld_cam_id


def solveRel3DCamPose(pts_corresp_dict, cam_params_dict, n_persons,
                      wrld_cam_id, M2_angle_thold=15,
                      verbose=False, save_dir=None):
    '''
    Solve relative 3D camera pose.
    '''
    _, Pts, BA_input, _ = solve3DPose(
        pts_corresp_dict, cam_params_dict, n_persons, wrld_cam_id=wrld_cam_id,
        M2_angle_thold=M2_angle_thold, verbose=verbose
    )
    Pts_BA, M2s_BA = buad.bundleAdjustmentWrapped(
        BA_input, max_iter=100, fix_cam_pose=False, verbose=verbose
    )
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    M2s_BA.insert(wrld_cam_id, M1)
    
    if save_dir is not None:
        save_file = os.path.join(save_dir, 'camera_pose_estimation.json')
        for i in range(len(M2s_BA)):
            M2s_BA[i] = M2s_BA[i].tolist()
        json.dump(M2s_BA, open(save_file, 'w'))
    return M2s_BA
