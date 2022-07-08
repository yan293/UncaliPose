"""
   This scipt defines functions for single-view tracking.
   Author: Yan Xu
   Date: April 10, 2022
"""

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from collections import defaultdict

import numpy as np


def matchSingleViewTwoFramesBoxes(
        boxfeat_frame1, boxfeat_frame2, method='person_id', verbose=False):
    '''
    Match single view bounding boxes cross frames, using Hungarian algorithm.
    '''
    boxfiles_frame1 = sorted(list(boxfeat_frame1.keys()))
    boxfiles_frame2 = sorted(list(boxfeat_frame2.keys()))
    
    def divideBoxfilesByCam(boxfiles):
        cam_to_boxes_dict = defaultdict(list)
        for boxfile in boxfiles:
            cam_id = boxfile.split('/')[-1].split('_')[0]
            cam_to_boxes_dict[cam_id].append(boxfile)
        return cam_to_boxes_dict
    
    cam_boxes_1 = divideBoxfilesByCam(boxfiles_frame1)
    cam_boxes_2 = divideBoxfilesByCam(boxfiles_frame2)
    
    box_matches = {}
    for cam in sorted(list(cam_boxes_1.keys())):
        
        if method == 'person_id':
            for boxfile_1 in cam_boxes_1[cam]:
                box_matches[boxfile_1] = None
                person_id =boxfile_1.split('/')[-1].split('_')[1]
                for boxfile_2 in cam_boxes_2[cam]:
                    if cam+'_'+person_id in boxfile_2:
                        box_matches[boxfile_1] = boxfile_2
                        break
            
        elif method == 'hungarian':
            for boxfile_1 in cam_boxes_1[cam]:
                box_matches[boxfile_1] = None
                
            if cam in cam_boxes_2.keys():
                feats_1 = [
                    boxfeat_frame1[boxfile] for boxfile in cam_boxes_1[cam]]
                feats_2 = [
                    boxfeat_frame2[boxfile] for boxfile in cam_boxes_2[cam]]
                dist_mat = distance.cdist(
                    np.array(feats_1), np.array(feats_2), metric='cosine')
                row_ind, col_ind = linear_sum_assignment(
                    dist_mat, maximize=False)
                
                # print(row_ind, col_ind)
            
                for i in range(len(row_ind)):
                    box_matches[cam_boxes_1[cam][row_ind[i]]] = \
                        cam_boxes_2[cam][col_ind[i]]
    if verbose:
        for key, val in box_matches.items():
            cam_1 = key.split('/')[-1].split('_')[0]
            person_1 = key.split('/')[-1].split('_')[1]
            cam_2 = val.split('/')[-1].split('_')[0]
            person_2 = val.split('/')[-1].split('_')[1]
            print(cam_1, person_1, '<->', cam_2, person_2)
    return box_matches


def singleViewTracking(boxfeat_list, method='person_id', verbose=False):
    '''
    Single-view multi-person tracking by matching the bounding box re-ID
    featurees of two consecutive frames.
    '''
    if len(boxfeat_list) == 1:
        track_box_feats = {}
        for key, val in boxfeat_list[0].items():
            track_box_feats[key] = [val]
        return track_box_feats
    
    num_frames = len(boxfeat_list)
    # step 1: match consecutive frames
    box_matches_over_time = []
    for i in range(num_frames - 1):
        box_matches_i = matchSingleViewTwoFramesBoxes(
            boxfeat_list[i], boxfeat_list[i+1], method=method,verbose=verbose)
        box_matches_over_time.append(box_matches_i)
    
    # step 2: track for each person of each view
    tracks, track_box_feats = {}, {}
    for person in box_matches_over_time[0].keys():
        tracks[person] = [person]
        track_box_feats[person] = [boxfeat_list[0][person]]
    
    for person in sorted(tracks.keys()):
        box_curr, box_next = person, box_matches_over_time[0][person]
        t_next = 1
        while (t_next < num_frames - 1) and (box_next is not None):
            tracks[person].append(box_next)
            track_box_feats[person].append(boxfeat_list[t_next][box_next])
            box_curr = box_next
            box_next = box_matches_over_time[t_next][box_curr]
            t_next += 1
        if box_next is not None:
            tracks[person].append(box_next)
            track_box_feats[person].append(boxfeat_list[t_next][box_next])
    
    return track_box_feats


def getTrackFeat(track_box_feats, method='mean'):
    '''
    Convert the track box features to a feature representing the whole track.
    '''
    track_feat = {}
    for key, val in track_box_feats.items():
        val = np.array(val)
        # for j in range(val.shape[0]):
        #     val[j] = val[j] / np.power(2, j)
        if method == 'mean':
            track_feat[key] = np.mean(val, axis=0)
        elif method == 'max':
            track_feat[key] = []
            for i in range(val.shape[1]):
                max_ind = np.argmax(np.abs(val[:, i]))
                track_feat[key].append(val[max_ind, i])
            track_feat[key] = np.array(track_feat[key])
        elif method == 'mean_with_sign_voting' or \
             method == 'max_with_sign_voting':
            track_feat[key] = []
            val_sign = np.sign(val).astype(int)
            for i in range(val.shape[1]):
                col_sign = val_sign[:, i]
                major_sign = np.argmax(np.bincount(col_sign + 1)) - 1
                selected_feat = val[:, i][col_sign==major_sign]
                if method == 'mean_with_sign_voting':
                    track_feat[key].append(np.mean(selected_feat))
                elif method == 'max_with_sign_voting':
                    track_feat[key].append(np.max(selected_feat))
            track_feat[key] = np.array(track_feat[key])
    return track_feat
