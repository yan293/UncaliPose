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


def _getRelPoseWrtCam(M2_dict, cam1_id):
    '''
    Compute the relative pose w.r.t. 'cam1_id'. 'direct' pose is computed
    using two cameras, 'indirect' poses are computed using third cameras.
    
    Input:
        M2_dict: {'M_1_0': [3x4], ...}, 'M_1_0' is the pose of cam 1 wrt cam0.
            cam1_id: Index of the chosen camera, from 0 to N-1.
    Output:
        M2s_wrt_cam1: A dictionary of M2s w.r.t the chosen camera 1.
            An example: {'M_1_0': {'direct': [3x4], 'indirect':[[3x4],...]},
            {...}, ...}. 'indirect' are the M2s computed using a third camera
    '''
    cam_set = set()
    for M2_name in M2_dict.keys():
        cam_set.add(int(M2_name.split('_')[-1]))
        cam_set.add(int(M2_name.split('_')[-2]))
    cam2_list = list(cam_set)
    cam2_list.remove(cam1_id)
    
    M2s_wrt_cam1 = {}
    for cam2_id in cam2_list:
        M2_direct = None
        if 'M_'+str(cam2_id)+'_'+str(cam1_id) in M2_dict:
            M2_direct = M2_dict['M_'+str(cam2_id)+'_'+str(cam1_id)]
            
        cam3_list = cam2_list.copy()
        cam3_list.remove(cam2_id)
        M2s_indirect = []
        for cam3_id in cam3_list:
            if 'M_'+str(cam2_id)+'_'+str(cam3_id) in M2_dict and \
               'M_'+str(cam3_id)+'_'+str(cam1_id) in M2_dict:  
                M23 = M2_dict['M_'+str(cam2_id)+'_'+str(cam3_id)]
                M31 = M2_dict['M_'+str(cam3_id)+'_'+str(cam1_id)]
                M21_ = b3dop.transmitExtrinMats(M23, M31)
                M2s_indirect.append(M21_)

        M2s_wrt_cam1['M_'+str(cam2_id)+'_'+str(cam1_id)] = {
            'direct': M2_direct, 'indirect': M2s_indirect}
    return M2s_wrt_cam1


def _selfValidateM2(M2_list):
    '''
    Camera pose self-validation, Find the best M2 closest to all other M2s.
    '''
    id_angle_pair = []
    for i in range(len(M2_list)):
        align_angles = []
        for j in range(len(M2_list)):
            if j == i:
                continue
            align_angles.append(
                b3dop.angleBtwnRotatMats(M2_list[i][:,:3], M2_list[j][:,:3]))
        id_angle_pair.append((i, np.mean(align_angles)))
    id_angle_pair = np.array(id_angle_pair)
    best_M2 = M2_list[int(id_angle_pair[id_angle_pair[:,1].argsort()[0]][0])]
    return best_M2


def selfValidateM2s(M2_dict, angle_thold=15):
    '''
    Perform camera pose self-validation for all cameras.
    
    Input:
        M2_dict: A dictionary of M2s, {'M_1_0':[3x4], ...},
            'M_1_0' means the pose of cam1 w.r.t. cam0.
        angle_thold: the angle sensitivity that dicides whether
            two matrixes are rotation-wise close.
    Output:
        valid_M2_dict
    '''
    # count number of cameras
    cam_ids = set()
    for key in M2_dict.keys():
        cam1_id = int(key.split('_')[1])
        cam2_id = int(key.split('_')[2])
        cam_ids.add(cam1_id)
        cam_ids.add(cam2_id)
    
    # select valid M2s for each camera
    valid_M2_dict = {}
    for wrld_cam_id in cam_ids:
        M2s_wrt_cam1 = _getRelPoseWrtCam(M2_dict, wrld_cam_id)
        for key, val in M2s_wrt_cam1.items():
            M2_list = list(val['indirect'])
            if val['direct'] is not None:
                M2_list = [val['direct']] + list(val['indirect'])
                
            best_M2 = _selfValidateM2(M2_list)
            # give "direct" estimation more trust
            if val['direct'] is not None:
                angle=b3dop.angleBtwnRotatMats(
                    val['direct'][:,:3], best_M2[:,:3])
                if angle < angle_thold:
                    best_M2 = val['direct']
            valid_M2_dict[key] = best_M2
    return valid_M2_dict


def solveMultiCamRelPose(pt_corresp_list, cam_param_list, M2_angle_thold=15):
    '''
    Estimate relative poses of all camera pairs.
    
    Input:
        pt_corresp_list: Point correspondence list [[Nx2], [Nx2], ....],
            invisible key points are represented as np.nan.
        cam_param_list: A list of cameras, each element of the list is a
            dictionary, containing necessary intrin/extrin
            information, an example of panoptic data would be,
            [{'name': ..., 'resolution': ..., 'K': [3x3],
            'distCoef': [1x4/1x5], 'R': [3x3], 't': [3x1]}, ...].
    Output:
        M2_dict: A dictionary of M2s: {'M_1_0':[3x4],...}, 'M_1_0'
            means the pose of camera 1 w.r.t. camera 0.
    '''
    wrld_cam_list = list(np.arange(len(cam_param_list)))
        
    M2_dict = {}
    for cam1_id in wrld_cam_list:
        cam1 = cam_param_list[cam1_id]
        K1, D1 = cam1['K'], cam1['distCoef']
        pts1 = pt_corresp_list[cam1_id]
        for cam2_id in range(len(cam_param_list)):
            if cam2_id == cam1_id:
                continue
            cam2 = cam_param_list[cam2_id]
            K2, D2 = cam2['K'], cam2['distCoef']
            pts2 = pt_corresp_list[cam2_id]
            pts1_, pts2_, _, _ = twov3d.countCovisiblePoints(pts1, pts2)
            if len(pts1_) != 0:
                _, _, M2 = twov3d.estRelativeCamPose(pts1_,pts2_,K1,D1,K2,D2)
                M2_dict['M_'+str(cam2_id)+'_'+str(cam1_id)] = M2
                
    # camera pose self-validation
    M2_dict_vaild = selfValidateM2s(M2_dict, angle_thold=M2_angle_thold)
    return M2_dict_vaild


def findBestCovisCamPairs(pt_corresp_list):
    '''
    For each camera, find the camera with most covisible keypoints.
    
    Input:
        pt_corresp_list: Point correspondence list [[Nx2], ....],
            invisible key points are represented as np.nan.
    Output:
        most_covis_pts_cam: A dictionary, where the key is a camera
            (idx) and the value is (best_cam2, # covis_pts).
    '''
    most_covis_pts_cam = {}
    n_cams = len(pt_corresp_list)
    
    for cam1 in range(n_cams - 1):
        pts1 = pt_corresp_list[cam1]
        covis_pts_count = 0
        best_cam = -1
        
        for cam2 in range(cam1 + 1, n_cams):
            pts2 = pt_corresp_list[cam2]
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


def selectRefCam(pt_corresp_list):
    '''
    Select the reference camera.  This camera should be the camera
    that has the most co-visible points with other cameras.
    '''
    most_covis_pts_cam = findBestCovisCamPairs(pt_corresp_list)
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


def merge3DPtsFromTwoCamPairs(Pts1, Pts2):
    '''
    Merge the 3D points triangulated from two different camera pairs.
    Assume the 3D points are aligned to the same camera coordinate system.
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


def merge3DJointsFromMultiCamPairs(Pts_list, M2_dict, n_joints):
    '''
    Merge all the 3D points in "Pts_list".
    
    Input:
        Pts_list: [[Nx3], ...], 3D points at different coordinate systems.
        M2_dict: A dictionary of M2s, {'M_1_0':[3x4], ...}, 'M_1_0' means
            the pose of cam1 w.r.t. cam0.
        n_joints: int, number of joints per person.
    Output:
        Pts1: [Nx3], merged 3D points at coordinate system "world_cam".
    '''
    n_person = len(Pts_list[0]) // n_joints
    
    # step 1: merge visible people one-by-one
    Pts1, Pts2_list = Pts_list[0], Pts_list[1:]
    for Pts2 in Pts2_list:
        person_joints = []
        for i in range(n_person):
            P1 = Pts1[i*n_joints:(i+1)*n_joints]
            P2 = Pts2[i*n_joints:(i+1)*n_joints]
            person_joints.append(merge3DPtsFromTwoCamPairs(P1, P2))
        Pts1 = np.concatenate(person_joints, axis=0)

    # step 2: merge invisible people all at once
    for Pts2 in Pts2_list:
        Pts1 = merge3DPtsFromTwoCamPairs(Pts1, Pts2)
        
    visible = []
    for i in range(n_person):
        Person = Pts1[i*n_joints:(i+1)*n_joints]
        visible.append(np.where(~np.isnan(Person[:, 0]))[0])
        
    return Pts1, visible


def triangulateFromAllCamPairs(
        pt_corresp_list, cam_param_list, M2_dict, ref_cam_id):
    '''
    Triangulate the 3D poses from all camera pairs.
    '''
    pt_l, cam_l = pt_corresp_list, cam_param_list
    Pts_all_cam_pairs = []
    
    for cam1_id in range(len(cam_param_list) - 1):
        for cam2_id in range(cam1_id+1, len(cam_param_list)):
            if cam2_id == ref_cam_id:
                cam1_id, cam2_id = cam2_id, cam1_id
            i1, i2 = cam1_id, cam2_id
            
            M1 = np.concatenate([np.eye(3), np.zeros((3,1))], axis=1)
            M2 = M2_dict['M_'+str(i2)+'_'+str(i1)]
            
            pts1, pts2, _, _ = twov3d.countCovisiblePoints(pt_l[i1], pt_l[i2])
            if len(pts1) == 0:
                continue
            
            Pts = twov3d.triangulatePoints(
                cam_l[i1]['K'], cam_l[i1]['distCoef'], M1,
                cam_l[i2]['K'], cam_l[i2]['distCoef'], M2,
                pt_l[i1], pt_l[i2])
            
            if i1 != ref_cam_id:
                Pts = b3dop.affineTransform3D(
                    Pts, M2_dict['M_'+str(ref_cam_id)+'_'+str(i1)])
                
            Pts_all_cam_pairs.append(Pts)

    return Pts_all_cam_pairs


def preparePoseEstData(pt_corresp_dict, cam_param_dict, n_person):
    '''
    Prepare data for 3D pose estimation.
    '''
    pt_corresp_list, cam_param_list, cam_names = [], [], []
    for cam_name in sorted(list(pt_corresp_dict.keys())):
        pt_corresp_list.append(pt_corresp_dict[cam_name]['keypoints'])
        cam_param_list.append(cam_param_dict[cam_name])
        cam_names.append(cam_name)
    n_joints = len(pt_corresp_list[0]) // n_person
    return pt_corresp_list, cam_param_list, cam_names, n_joints


def prepareBundleAdjustmentData(
        Pts, pt_corresp_list, cam_param_list, M2_dict, ref_cam_id):
    '''
    Prepare data for Bundle Adjustment.
    '''
    K2s, D2s, M2s, p2s = [], [], [], []
    for i, cam_params in enumerate(cam_param_list):
        if i == ref_cam_id:
            p1 = pt_corresp_list[i]
            K1 = cam_params['K']
            D1 = cam_params['distCoef']
            M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            continue
        p2s.append(pt_corresp_list[i])
        K2s.append(cam_params['K'])
        D2s.append(cam_params['distCoef'])
        
        _, _, _, n_covis = twov3d.countCovisiblePoints(
            Pts[:,:2], pt_corresp_list[i])
        if n_covis >= 30:   
            M2 = b3dop.solveAbsCamPose(
              Pts,pt_corresp_list[i],cam_params['K'],D=cam_params['distCoef'])
        else:
            M2 = M2_dict['M_'+str(i)+'_'+str(ref_cam_id)]
        M2s.append(M2)
        
    BA_input = {
        'p1': p1, 'p2s': p2s, 'P': Pts,
        'K1': K1, 'D1': D1, 'M1': M1,
        'K2s': K2s, 'D2s': D2s, 'M2s': M2s}
    return BA_input


def solveMultiView3DHumanPoses(
        pt_corresp_dict, cam_param_dict, n_person, wrld_cam_id=None,
        M2_angle_thold=15, Pts_prev=None, verbose=True):
    '''
    Solve multi-view multi-person 3D human poses and relative camera poses.
    '''
    if verbose:
        print("\nSolve 3D human and camera poses:\n====================")
        
    pt_corresp_list, cam_param_list, cam_names, n_joints = preparePoseEstData(
        pt_corresp_dict, cam_param_dict, n_person)
    
    # Step 1: Solve relative camera poses using self-validation
    M2_dict = solveMultiCamRelPose(
        pt_corresp_list, cam_param_list, M2_angle_thold=M2_angle_thold)
    
    # Step 2: Select reference camera
    if wrld_cam_id is None:
        wrld_cam_id = selectRefCam(pt_corresp_list)
    
    # Step 3: Triangulate each person's 3D pose
    Pts_all_cam_pairs = triangulateFromAllCamPairs(
        pt_corresp_list, cam_param_list, M2_dict, wrld_cam_id)
    
    # Step 4: Merge the 3D pose of all people
    Pts, _ =merge3DJointsFromMultiCamPairs(Pts_all_cam_pairs,M2_dict,n_joints)
    
    # Step 5: Prepare data for Bundle Adjustment
    BA_input = prepareBundleAdjustmentData(
        Pts, pt_corresp_list, cam_param_list, M2_dict, wrld_cam_id)
    
    if verbose:
        print("Done.\nRef cam id:{}, name:{}\n====================\n".format(
            wrld_cam_id, cam_names[wrld_cam_id]))

    return Pts, BA_input, cam_names[wrld_cam_id]
