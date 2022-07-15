#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This scipt defines Panoptic class for the Panoptic dataset.
   Author: Yan Xu
   Date: Jan 15, 2022
"""
import sys
sys.path.append('./..')
from .. import basic_3d_operations as b3dops
from .. import box_processing as bp
from .. import tracking as tk
from collections import defaultdict
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


class Shelf(object):
    
    def __init__(self, data_dir, config=None):
        if data_dir[-1] == '/': data_dir = data_dir[:-1]
        self.data_name = data_dir.split('/')[-1]
        self.data_dir = data_dir
        self.video_frame_dir = os.path.join(data_dir, 'video_frame')
        self.pose3d_file_dir = os.path.join(data_dir, 'pose3d_label')
        self.pose2d_file_dir = os.path.join(data_dir, 'pose2d_label')
        self.boxcrop_dir = os.path.join(data_dir, 'box_crop')
        self.calibration_file = os.path.join(data_dir,'calibration.json')
        self.num_cam = 5
        
        if not os.path.exists(self.calibration_file):
            self.genCaliFileFromRawData()
        self.cam_params_dict = self._loadCalibrationParameters()
        
        self.joints_ids = np.arange(14)
        self.body_edges = np.array(
            [[2,8],[3,9],[2,3],[12,13],[0,1],[1,2],[3,4],
             [4,5],[6,7],[7,8],[8,12],[9,10],[10,11],[9,12]])
        
        self.config = config

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

    def genPoseLabelFilesFromRawData(self):
        '''
        Generate 2D&3D pose label files of self-defined formate from Raw data.
        '''
        actors_gt_file = os.path.join(self.data_dir, 'raw_data/actorsGT.mat')
        actors_gt = io.loadmat(actors_gt_file)
        actor_2d_gt = actors_gt['actor2D'][0]
        actor_3d_gt = actors_gt['actor3D'][0]
        n_persons = len(actor_2d_gt)
        n_cameras = actor_2d_gt[0][0][0].shape[1]
        n_frames = len(actor_2d_gt[0])

        # --- save 2D human pose label
        for cam_id in range(n_cameras):
            for i_f in range(n_frames):
                bodies = []
                for i_p in range(n_persons):
                    joints_2d = np.array(actor_2d_gt[i_p][i_f][0])[0][cam_id]
                    if joints_2d.shape[1] == 0: continue
                    bodies.append({'id': i_p, 'joints': joints_2d.T.tolist()})
                if len(bodies) == 0: continue
                save_dir = os.path.join(self.pose2d_file_dir, 'Camera'+str(cam_id))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file = os.path.join(save_dir, str(i_f).zfill(8)+'.json')
                save_data = {'joint_type': 'Shelf', 'bodies': bodies}
                json.dump(save_data, open(save_file, 'w'))
        print('Generated 2D pose label file: {}.'.format(save_file))

        # --- save 3D human pose label
        for i_f in range(n_frames):
            bodies = []
            for i_p in range(n_persons):
                joints_3d = np.array(actor_3d_gt[i_p][i_f][0])
                if joints_3d.shape[1] == 0: continue
                bodies.append({'id': i_p, 'joints': joints_3d.T.tolist()})
            if len(bodies) == 0: continue
            save_dir = self.pose3d_file_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = os.path.join(save_dir, str(i_f).zfill(8)+'.json')
            save_data = {'joint_type': 'Shelf', 'bodies': bodies}
            json.dump(save_data, open(save_file, 'w'))
        print('Generated 3D pose label file: {}.\n'.format(save_file))
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
    
    def getSingleFrameMultiViewBoxes(
            self, frame_id, box_joints_margin=1.2, box_ios_thold=0.7,
            box_size_thold=(20, 20),joints_inside_img_ratio=0.6,
            box_inside_img_ratio=0.6, img_postfix='.jpg', verbose=True,
            resize=(128, 256), replace_old=False):
        '''
        Crop the bounding boxes from myltiple views for a video frame.
        '''
        if self.config is not None:
            box_config = self.config['boxprocessing']
            joints_inside_img_ratio = box_config['joints_inside_img_ratio']
            box_inside_img_ratio = box_config['box_inside_img_ratio']
            box_joints_margin = box_config['box_joints_margin']
            box_size_thold = box_config['box_size_thold']
            box_ios_thold = box_config['box_ios_thold']
            replace_old = box_config['replace_old']
            resize = box_config['resize']
            
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
                person_id = ids[-1]
                box = bp.cutBoxAroundJoints(
                    im_size, joints, margin_ratio=box_joints_margin)
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
            # idxes = np.arange(len(boxes0))[idxes1][idxes3][idxes4]
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
        if self.config is not None:
            reid_config = self.config['reid']
            num_prev_frames = reid_config['num_prev_frames']
            trk_feat_method = reid_config['trk_feat_method']
            trking_method = reid_config['trking_method']
            
        # --- current frame reid feature
        frame_box_dir = os.path.join(
            self.boxcrop_dir,'frame'+str(frame_id).zfill(8))
        reid_feat_file = os.path.join(frame_box_dir, 'box_reid_feat.pkl')
        if not os.path.exists(reid_feat_file):
            bp.extractBoxReIDFeature(
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
    
    def genPtsCorrepFromBoxClus(
            self, boxfile_clusters, verbose=True, noise_sz=None):
        '''
        Get 2D-2D point correspondences from box crop files clusters.
        '''
        if self.config is not None:
            correp_config = self.config['correspondence']
            noise_sz = correp_config['noise_sz']
        
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
            if noise_sz is not None:
                noise = np.random.uniform(low=-noise_sz, high=noise_sz,
                    size=corresp_dict[cam_id]['keypoints'].shape)
                corresp_dict[cam_id]['keypoints'] += noise

        # save point correspondences
        save_file = os.path.join(frame_box_dir, 'box_gen_pt_corr.pkl')
        pkl.dump(corresp_dict, open(save_file, 'wb'))
        n_persons = len(persons)
        if verbose:
            print('\nSaved to:\n\"{}\".'.format(save_file))
            print('====================\n')
        return corresp_dict, n_persons
    
    def getMultiFramePtCorrsps(self, frame_list):
        '''
        Get point correpondences from multiply frames.
        '''
        n_person_all = 0
        pt_corrs_multi_frame = defaultdict(list)
        for frame_id in frame_list:
            pt_corrs_frame_file = os.path.join(
                self.boxcrop_dir, 'frame' + str(frame_id).zfill(8),
                'box_gen_pt_corr.pkl')
            pt_corrs_single_frame = pkl.load(open(pt_corrs_frame_file, 'rb'))
            n_person_all += len(list(pt_corrs_single_frame.keys()))
            for person_id, val in pt_corrs_single_frame.items():
                pt_corrs_multi_frame[person_id].append(val)
                
        for person_id, val_list in pt_corrs_multi_frame.items():
            keypts, boxfile_list = [], []
            for val in val_list:
                keypts.append(val['keypoints'])
                boxfile_list += val['box_files']
            pt_corrs_multi_frame[person_id] = {
                'keypoints': np.concatenate(keypts, axis=0),
                'box_files': boxfile_list
            }
        return pt_corrs_multi_frame, n_person_all
    
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
        # --- Step 1: load reference camera parameters
        R = self.cam_params_dict[ref_cam_name]['R']
        t = np.array(self.cam_params_dict[ref_cam_name]['t']).squeeze()
        
        # --- Step 2: load ground truth
        pose3d_gt_dict = json.load(open(os.path.join(
            self.pose3d_file_dir, str(frame_id).zfill(8) + '.json'), 'r'))
        Pts_by_person, person_ids = {}, []
        for body_3d in pose3d_gt_dict['bodies']:
            joints3d_gt = np.array(body_3d['joints']).T
            joints3d_gt = np.array(joints3d_gt + np.linalg.pinv(R).dot(t))
            Pts_by_person[int(body_3d['id'])] = {'pose3d_GT': joints3d_gt}
        
        # --- Step 3: convert estimated relative 3D pose to absolut 3D pose
        assert np.array(Pts_est_rela).shape[1] == 3
        Pts_est_world = np.nan * np.ones_like(Pts_est_rela)
        idx = np.arange(len(Pts_est_rela))[~np.isnan(Pts_est_rela[:, 0])]
        Pts_est_world[idx] = np.linalg.pinv(R).dot(Pts_est_rela[idx].T).T
        
        # --- Step 4: solve scale ambiguity for quantitative evaluation
        n_joints = len(self.joints_ids)
        Pts_gt, Pts_est = [], []
        for k, person_id in enumerate(sorted(list(Pts_by_person.keys()))):
            Jnts_gt = Pts_by_person[person_id]['pose3d_GT']
            Jnts_est = Pts_est_world[k*n_joints:(k+1)*n_joints]
            
            for edge in self.body_edges:
                i, j = edge[0], edge[1]
                if ~np.isnan(Jnts_est[i][0]) and ~np.isnan(Jnts_est[j][0]):
                    break
            bone_len_gt = np.linalg.norm(Jnts_gt[i][0] - Jnts_gt[j][0])
            bone_len_est = np.linalg.norm(Jnts_est[i][0] - Jnts_est[j][0])
            scale = bone_len_gt / bone_len_est
            Jnts_est = Jnts_est * scale
            Jnts_est = Jnts_est - Jnts_est[0, :] + Jnts_gt[0, :]
            
            Pts_by_person[person_id]['pose3d_Est'] = Jnts_est
            Pts_gt.append(Jnts_gt)
            Pts_est.append(Jnts_est)
        Pts_gt = np.concatenate(Pts_gt, axis=0)
        Pts_est = np.concatenate(Pts_est, axis=0)
        
        return Pts_gt, Pts_est, Pts_by_person, scale
    