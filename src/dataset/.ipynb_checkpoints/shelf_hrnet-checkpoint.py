#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This scipt defines Panoptic class for the Panoptic dataset.
   Author: Yan Xu
   Date: Feb 06, 2022
"""
import sys
sys.path.append('./..')
from .. import basic_3d_operations as b3dops
from .. import box_processing as bp
from .. import box_clustering as bc
from .. import tracking as tk
from scipy import io
import pickle as pkl
import numpy as np
import shutil
import glob
import json
import copy
import time
import cv2
import os


class Shelf_HRNet(object):
    
    def __init__(self, data_dir):
        if data_dir[-1] == '/': data_dir = data_dir[:-1]
        self.data_name = data_dir.split('/')[-1]
        self.data_dir = data_dir
        self.video_frame_dir = os.path.join(data_dir, 'video_frame')
        self.calibration_file = os.path.join(data_dir,'calibration.json')
        self.pose3d_file_dir = os.path.join(data_dir, 'pose3d_label')
        self.pose2d_file_dir = os.path.join(data_dir, 'pose2d_label_hrnet')
        self.hrnet_det_file = os.path.join(data_dir, 'hrnet_coco.pkl')
        self.boxcrop_dir = os.path.join(data_dir, 'box_crop_hrnet')
        
        if not os.path.exists(self.calibration_file):
            self.genCaliFileFromRawData()
        self.cam_params_dict = self._loadCalibrationParameters()
        self.num_cam = 5
        
        self.joints_ids = np.array([0]+list(range(5,17))) # nose + body joints
        self.body_edges = [[0,1],[0,2],[1,2],[1,3],[1,7],[2,4],[2,8],
            [3,5],[4,6],[7,8],[7,9],[8,10],[9,11],[10,12]]

    def _loadCalibrationParameters(self):
        '''
        Load camera parameters and convert to numpy,
        since data were written in string.
        '''
        cameras_raw = json.load(open(self.calibration_file, 'r'))
        cameras = {}
        for cam_id, cam_params in cameras_raw.items():
            cameras[int(cam_id)] = {
                'P': np.matrix(cam_params['P']),
                'K': np.matrix(cam_params['K']),
                'distCoef': np.array(cam_params['distCoef']),
                'M': np.array(cam_params['M']),
                'R': np.matrix(cam_params['R']),
                't': np.array(cam_params['t']).reshape((3,1)),
                'resolution': np.array(cam_params['resolution']),
            }
        return cameras
    
    def getRelativeCameraPose(self, ref_cam_name):
        '''
        Convert the absolute camera pose to relative camera pose.
        '''
        cam_ids = sorted(self.cam_params_dict)
        M10 = self.cam_params_dict[ref_cam_name]['M']
        M01 = b3dops.invertExtrinMat(M10)
        M2s_rel = []
        for cam_id in cam_ids:
            # if cam_id == ref_cam_name: continue
            M20 = self.cam_params_dict[cam_id]['M']
            M21 = b3dops.transmitExtrinMats(M20, M01)
            M2s_rel.append(M21)
        return M2s_rel

    def genHRNetPoseLabelFiles(self, conf_thold=0.5):
        '''
        Generate 2D&3D pose label files from HRNet detection.
        '''
        pose_hrnet = pkl.load(open(self.hrnet_det_file, 'rb'))
        keys = sorted(list(pose_hrnet.keys()))
        
        for key in keys:
            cam_id, frame_id = int(key.split('_')[0]), int(key.split('_')[1])
            joints_list = pose_hrnet[key][:4]  # at most 4 people for Shelf
            
            bodies = []
            for person_id, joints_dict in enumerate(joints_list):
                joints = joints_dict['pred']
                confidence = joints[:, 2]
                joints[confidence < conf_thold] = np.nan
                bodies.append(
                    {'id': person_id, 'joints': joints.T.tolist()})
                
            if len(bodies) == 0: continue
            save_dir = os.path.join(self.pose2d_file_dir,'Camera'+str(cam_id))
            if not os.path.exists(save_dir): os.makedirs(save_dir)
                
            save_file = os.path.join(save_dir, str(frame_id).zfill(8)+'.json')
            save_data = {'joint_type': 'Shelf_HRNet', 'bodies': bodies}
            json.dump(save_data, open(save_file, 'w'))
        print('Saved HRNet 2D pose label to {}.'.format(self.pose2d_file_dir))
        return None
    
    def genCaliFileFromRawData(self):
        '''
        Generate calibration file of self-defined format from raw data.
        '''
        cali_file_dir = os.path.join(self.data_dir, 'raw_data/Calibration')
        cali_files = sorted(glob.glob(os.path.join(cali_file_dir, '*.cal')))
        cam_params_dict = {}

        for cam_id, cali_file in enumerate(cali_files):
            cam_params = []
            with open(cali_file) as f:
                lines = f.readlines()
                for line in lines:
                    cam_params.append([float(x) for x in line.split()])

            im_sz = np.array(cam_params[3]).astype(int)
            P = np.stack(np.array(cam_params[:3]))
            K = np.stack(cam_params[4:7])
            R = np.stack(cam_params[7:10])
            t = np.asarray(cam_params[10])
            M = np.concatenate((R, np.expand_dims(t, 1)), axis=1)
            M[0], M[1] = -M[0], -M[1]    # only this can make P = K @ M!!!
            R_, t_ = M[:, :3], M[:, 3]
            D = np.zeros(5)
            cam_params_dict[cam_id] = {
                'name': 'Camera' + str(cam_id),
                'resolution': im_sz.tolist(),
                'P': P.tolist(),
                'K': K.tolist(),
                'distCoef': D.tolist(),
                'M': M.tolist(),
                'R': R_.tolist(),
                't': t_.tolist(),
            }
        save_file = os.path.join(self.data_dir, 'calibration.json')
        json.dump(cam_params_dict, open(save_file, 'w'))
        print('Generates calibration file: {}.\n'.format(save_file))
        return cam_params_dict
    
    def fetchVideoFrameFile(self, cam_name, frame_id):
        return os.path.join(
            self.video_frame_dir, cam_name+'/{0:08}.png'.format(frame_id))
    
    def getSingleFrameMultiView2DJoints(self, frame_id):
        '''
        Extract single frame multi-view joints.
        '''
        if self.pose2d_file_dir[-1] == '/':
            self.pose2d_file_dir = self.pose2d_file_dir[:-1]
        dataset_name = self.pose2d_file_dir.split('/')[-2]
        camera_folder_list = glob.glob(os.path.join(self.pose2d_file_dir,'*'))
        
        joints_dict = {}
        for camera_folder in sorted(camera_folder_list):
            cam_name = camera_folder.split('/')[-1]
            pose2d_json_file = os.path.join(
                self.pose2d_file_dir,cam_name,str(frame_id).zfill(8)+'.json')
            
            if os.path.exists(pose2d_json_file):
                frame_joints = json.load(open(pose2d_json_file, 'r'))
                frame_joints_dict = {}
                
                for body in frame_joints['bodies']:
                    person_id = body['id']
                    joints_2d = np.array(body['joints'])[:, self.joints_ids]
                    frame_joints_dict[(frame_id, person_id)] = joints_2d
                joints_dict[cam_name] = frame_joints_dict
            else:
                joints_dict[cam_name] = {}
        return joints_dict
    
    def getSingleFrameMultiViewBoxes(self, frame_id,
                                      box_joints_margin=1.2,
                                      box_ios_thold=0.7,
                                      box_size_thold=(20, 20),
                                      joints_vis_ratio=0.6,
                                      joints_inside_img_ratio=0.6,
                                      box_inside_img_ratio=0.6,
                                      img_postfix='.jpg',
                                      verbose=True,
                                      resize=(128, 256),
                                      replace_old=False):
        '''
        Crop the bounding boxes from myltiple views for a video frame.
        '''
        time_start = time.time()
        save_crop_dir = os.path.join(
            self.boxcrop_dir, 'frame' + str(frame_id).zfill(8))
            
        box_joint_map_file = os.path.join(save_crop_dir, 'box_joints_map.pkl')
        if os.path.exists(box_joint_map_file) and not replace_old:
            return save_crop_dir
        
        if not os.path.exists(save_crop_dir):
            os.makedirs(save_crop_dir)
        else:
            old_files = glob.glob(os.path.join(save_crop_dir,'*'+img_postfix))
            for img_file in old_files: os.remove(img_file)
            reid_feat_file = os.path.join(save_crop_dir,'box_reid_feat.pkl')
            if os.path.exists(reid_feat_file): os.remove(reid_feat_file)
        
        joints_dict = self.getSingleFrameMultiView2DJoints(frame_id)
        box_joints_map = {}
        
        if verbose:
            print('Get bounding box crops\n=============')
        for camera_name, camera_joints_dict in joints_dict.items():
            if verbose:
                print('{} | camera {} | frame {}.'.format(
                    self.data_name, camera_name, frame_id))
            image_file = self.fetchVideoFrameFile(camera_name, frame_id)
            im = cv2.imread(image_file)
            im_size = im.shape[0], im.shape[1]
            
            # For a camera, prepare boxes and prefixes of file name
            boxes0, prefixes, joints_list = [], [], []
            for ids, joints in camera_joints_dict.items():
                vis_ratio = np.sum(~np.isnan(joints[0]))/float(len(joints[0]))
                if vis_ratio < joints_vis_ratio: continue
                person_id = ids[-1]
                box = bp.cutBoxAroundJoints(
                    im_size, joints[:2], margin_ratio=box_joints_margin)
                boxes0.append(box)
                prefixes.append(
                    camera_name.replace('_', '-') + '_' + str(person_id))
                _,_,joints_vis = bp.countNumJointsInsideImage(im_size, joints)
                joints_list.append(joints_vis)
                
            # box selection, "removeBlockedBoxes" must be placed at beginning
            # since, in this function, boxes will impact each other.
            boxes, idxes1 = bp.removeBlockedBoxes(
                boxes0, box_ios_thold=box_ios_thold)
            boxes, idxes2 = bp.removeOutsideViewJoints(
                boxes, joints_inside_img_ratio=joints_inside_img_ratio)
            boxes, idxes3 = bp.removeOutsideViewBoxes(
                boxes, box_inside_img_ratio=box_inside_img_ratio)
            boxes, idxes4 = bp.removeSmallBoxes(
                boxes, box_size_thold=box_size_thold)
            
            idxes = np.arange(len(boxes0))[idxes1][idxes2][idxes3][idxes4]
            prefixes = [prefixes[i] for i in idxes]
            joints_list = [joints_list[i] for i in idxes]
                
            # crop boxes and save
            boxcrops, boxfiles = bp.cropBoxesInImage(
                image_file, boxes, save_dir=save_crop_dir, prefixes=prefixes,
                img_postfix=img_postfix, resize=resize)
            
            # save joints to box map
            for ii in range(len(boxfiles)):
                box_joints_map[boxfiles[ii].split('/')[-1]] = joints_list[ii]
                
        pkl.dump(box_joints_map, open(box_joint_map_file, "wb"))
        if verbose:
            print('Box-joints map saved to \"{}\".\n[{:.2f} seconds]'.format(
                box_joint_map_file, time.time() - time_start))
            print('=============\n')
        return save_crop_dir
    
    def getFrameReIDFeat(
            self, frame_id, num_prev_frames=0, reid_model=None,
            trking_method='person_id', trk_feat_method='mean',
            reid_log_file=None
    ):
        '''
        Get the bounding box ReID features of a given frame.
        '''
        # --- current frame reid feature
        frame_box_dir = os.path.join(
            self.boxcrop_dir,'frame'+str(frame_id).zfill(8))
        reid_feat_file = os.path.join(frame_box_dir, 'box_reid_feat.pkl')
        if not os.path.exists(reid_feat_file):
            bc.extractBoxReIDFeature(
                frame_box_dir, reid_model=reid_model, log_file=reid_log_file)
            # time.sleep(10)
        if not os.path.exists(reid_feat_file):
            print('Re-ID feature file not found!')
            return None
        reid_feat = pkl.load(open(reid_feat_file, 'rb'))
        
        # --- previous frames reid feature
        if num_prev_frames > 0:
            box_feats = [reid_feat]
            for i in range(num_prev_frames):
                frame_prev_id = frame_id - (i + 1)
                prev_reid_feat_file = os.path.join(
                    self.boxcrop_dir,'frame'+str(frame_prev_id).zfill(8),
                    'box_reid_feat.pkl')
                if not os.path.exists(prev_reid_feat_file):
                    continue
                prev_reid_feat = pkl.load(open(prev_reid_feat_file,'rb'))
                box_feats.append(prev_reid_feat)
            
            # tracking and track feature representation
            track_box_feats = tk.singleViewTracking(
                box_feats, method=trking_method, verbose=False)
            track_feat = tk.getTrackFeat(
                track_box_feats, method=trk_feat_method)
            reid_feat = track_feat

        return reid_feat
    
    def genPtsCorrepFromBoxClus(self, boxfile_clusters, verbose=True):
        '''
        Get 2D-2D point correspondences from box crop files clusters.
        '''
        if verbose:
            print("\nGet 2D-2D correspondences:\n====================")
            
        # --- Step 1: get parameters (num_cams, person_ids...)
        values = list(boxfile_clusters.values())
        while values[0] is None or len(values[0]) == 0: values.pop(0)
        frame_box_dir = os.path.join(
            self.boxcrop_dir, values[0][0].split('/')[-2])
        persons, cam_ids = set(), set()
        for person, boxfile_list in boxfile_clusters.items():
            persons.add(person)
            for boxfile in boxfile_list:
                cam_ids.add(int(boxfile.split('/')[-1].split('_')[0][6:]))
        cam_ids, persons = sorted(list(cam_ids)), sorted(list(persons))

        # Step 2: Establish 2D-2D point correpondences
        box_joints_map = pkl.load(
            open(os.path.join(frame_box_dir, 'box_joints_map.pkl'), 'rb'))
        
        corresp_dict = {}
        for cam_id in cam_ids:
            corresp_dict[cam_id] = {'keypoints': [], 'box_files': []}

        for person in persons:
            if verbose:
                print('Get pt correspondences of person {}.'.format(person))
            
            # set all people to np.nan, if visible, value will change later
            for cam_id in cam_ids:
                corresp_dict[cam_id]['keypoints'].append(
                    np.ones((len(self.joints_ids), 2)) * np.nan)
                corresp_dict[cam_id]['box_files'].append(None)

            for boxfile in boxfile_clusters[person]:
                cam_id = int(boxfile.split('/')[-1].split('_')[0][6:])
                joints_2d = box_joints_map[boxfile.split('/')[-1]][:2].T
                assert len(joints_2d) == len(self.joints_ids)
                corresp_dict[cam_id]['keypoints'][-1] = joints_2d
                corresp_dict[cam_id]['box_files'][-1] = boxfile
                
        for cam_id in cam_ids:
            corresp_dict[cam_id]['keypoints'] = np.concatenate(
                corresp_dict[cam_id]['keypoints'], axis=0)

        # save point correspondences
        save_file = os.path.join(frame_box_dir, 'box_gen_pt_corr.pkl')
        pkl.dump(corresp_dict, open(save_file, 'wb'))
        n_persons = len(persons)
        if verbose:
            print('\nSaved to:\n\"{}\".'.format(save_file))
            print('====================\n')
        return corresp_dict, n_persons
    
    def convertToWrldCoord(self, Pts, cam_name):
        '''
        Convert 3D coordinates w.r.t. a camera to the world coordinate system.
        "Pts" should be in the shape of (N, 3).
        '''
        assert np.array(Pts).shape[1] == 3
        R = self.cam_params_dict[cam_name]['R']
        t = np.array(self.cam_params_dict[cam_name]['t']).squeeze()
        Pts_w = np.nan * np.ones_like(Pts)
        idx = np.arange(len(Pts))[~np.isnan(Pts[:, 0])]
        Pts_w[idx] = np.linalg.pinv(R).dot((Pts[idx] - t).T).T
        return Pts_w
    
    def alignEstAndGTCoords(self, frame_id, Pts_est_rela, ref_cam_name):
        '''
        Align estimated and the ground truth 3D poses.
        '''
        idx_gt_to_coco = np.array([13,8,9,7,10,6,11,2,3,1,4,0,5])

        # --- Step 1: load reference camera parameters
        R = self.cam_params_dict[ref_cam_name]['R']
        t = np.array(self.cam_params_dict[ref_cam_name]['t']).squeeze()

        # # --- Step 2: load ground truth
        # pose3d_gt_dict = json.load(open(os.path.join(
        #     self.pose3d_file_dir, str(frame_id).zfill(8) + '.json'), 'r'))
        # Pts_gt = []
        # for body_3d in pose3d_gt_dict['bodies']:
        #     joints3d_gt = np.array(body_3d['joints']).T
        #     joints3d_gt = np.array(joints3d_gt + np.linalg.pinv(R).dot(t))
        #     Pts_gt.append(joints3d_gt[idx_gt_to_coco])
        # Pts_gt = np.concatenate(Pts_gt, axis=0)

        # --- Step 3: convert estimated relative 3D pose to absolut 3D pose
        assert np.array(Pts_est_rela).shape[1] == 3
        Pts_est_world = np.nan * np.ones_like(Pts_est_rela)
        idx = np.arange(len(Pts_est_rela))[~np.isnan(Pts_est_rela[:, 0])]
        Pts_est_world[idx] = np.linalg.pinv(R).dot(Pts_est_rela[idx].T).T
        
        # --- Step 4: align the GT and est human poses
        
        return None, Pts_est_world, None
    