"""
   This scipt contains functions for visualization.
   Author: Yan Xu
   Date: Jan 31, 2022
"""

import numpy as np
import copy


def computePosesDist(Joints_3d_1, Joints_3d_2):
    '''
    Compute the distance between two 3D poses.
    '''
    body_weight_1 = np.mean(Joints_3d_1, axis=0)
    body_weight_2 = np.mean(Joints_3d_2, axis=0)
    dist = np.linalg.norm(body_weight_1 - body_weight_2)
    return dist


def findClosetGTPose(Joints_3d_gt_list, Joints_3d_est_list):
    '''
    Assign the est poses with the closest ground truth poses.
    (In Shelf and Campus, GTs are less than ESTs from HRNet.)
    '''
    closest_ests = []
    all_ests = set(range(len(Joints_3d_est_list)))
    for Joints_3d_gt in Joints_3d_gt_list:
        dists = []
        left_ests = list(all_ests - set(closest_ests))
        for i in left_ests:
            dists.append(computePosesDist(Joints_3d_gt,Joints_3d_est_list[i]))
        np.argmin(dists)[0][0]
        
    return None


def countCorrectEstBodyParts(Joints_3d_gt, Joints_3d_est, body_edges,
                             alpha=0.5):
    '''
    Count the number of correctly estimated body parts.
    
    Input:
        Joints_3d_gt, [Nx3], ground truth 3D human pose (multiple people).
        Joints_3d_est, [Nx3], estimated 3D human pose (multiple people).
        body_edges, body edges (connections of body joints indexs).
        alpha, threshold.
    Output:
        total_count, count of all body parts.
        correct_count, count of correctly estimated body parts.
    '''
    total_count = len(body_edges)
    correct_count = 0
    for edge in body_edges:
        i, j = edge[0], edge[1]
        s0, e0 = Joints_3d_gt[i], Joints_3d_gt[j]
        s1, e1 = Joints_3d_est[i], Joints_3d_est[j]
        e1 = s0 + (e1 - s0)
        s1 = s0
        displace = (np.linalg.norm(s0 - s1) + np.linalg.norm(e0 - e1)) / 2.
        part_len = np.linalg.norm(s0 - e0)
        if displace <= alpha * part_len:
            correct_count += 1
    return total_count, correct_count


def computeMPJPE(Joints_3d_gt, Joints_3d_est):
    '''
    Compute "Mean Per Joint Position Error".
    '''
    idx1 = ~np.isnan(Joints_3d_gt[:, 0])
    idx2 = ~np.isnan(Joints_3d_est[:, 0])
    idx = np.logical_and(idx1, idx2)
    Joints_3d_gt_vis = Joints_3d_gt[idx]
    Joints_3d_est_vis = Joints_3d_est[idx]
    n_vis_joints = len(Joints_3d_gt_vis)
    mpjpe = np.mean(np.linalg.norm(Joints_3d_gt_vis-Joints_3d_est_vis,axis=1))
    return n_vis_joints, mpjpe


def countNumCorrectPeople(Joints_3d_gt, Joints_3d_est, body_edges, n_person,
                          alpha=0.5, parts_correct_ratio=0.6):
    '''
    Count number of correctly reconstructed people.
    '''
    n_joints = len(Joints_3d_gt) // n_person
    n_correct = 0
    for i in range(n_person):
        start, end = i * n_joints, (i + 1) * n_joints
        body3d_gt = Joints_3d_gt[start:end]
        body3d_est = Joints_3d_est[start:end]
        total_count, correct_count = countCorrectEstBodyParts(
            body3d_gt, body3d_est, body_edges, alpha=alpha)
        
        if float(correct_count) / float(total_count) >= parts_correct_ratio:
            n_correct += 1
    return n_person, n_correct


