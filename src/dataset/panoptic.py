#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This scipt defines Panoptic class for the Panoptic dataset.
   Author: Yan Xu
   Update: Nov 29, 2021
"""

import sys
sys.path.append('./..')
from .. import basic_3d_operations as b3dops
from .. import box_processing as bp
import pickle as pkl
import numpy as np
import shutil
import glob
import json
import copy
import time
import cv2
import os


class Panoptic(object):
    
    def __init__(self, data_dir, config=None):
        if data_dir[-1] == '/': data_dir = data_dir[:-1]
        self.data_dir = data_dir
        self.data_name = data_dir.split('/')[-1]
        self.video_frame_dir = os.path.join(data_dir, 'video_frame')
        self.pose3d_file_dir = os.path.join(data_dir, 'pose3d_label')
        self.pose2d_file_dir = os.path.join(data_dir, 'pose2d_label')
        self.boxcrop_dir = os.path.join(data_dir, 'box_crop')
        self.calibration_file = os.path.join(data_dir,'calibration.json')
        
        self.cam_list = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]
        self.num_cam = len(self.cam_list)
        self.cam_params_dict = self._loadCalibrationParameters()
        
        self.joints_ids = np.arange(15)  # COCO 19 w/o face
        self.body_edges = np.array([[0,1],[0,3],[3,4],[4,5],[0,2],[2,6],[6,7],
            [7,8],[2,12],[12,13],[13,14],[0,9],[9,10],[10,11]])
        
        self.config = config

    def _loadCalibrationParameters(self):
        calib = json.load(open(self.calibration_file, 'r'))
        cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}
        selected_cameras = {}
        for key in self.cam_list:
            cam = cameras[key]
            K = np.matrix(cam['K'])
            D = np.array(cam['distCoef'])
            R = np.matrix(cam['R'])
            t = np.array(cam['t']).squeeze()
            M = np.array(np.concatenate((R, np.expand_dims(t,axis=1)),axis=1))
            # # --- change (x, y) to (-x, -y), the direction of z won't change
            # R[0], R[1] = -R[0], -R[1]
            # t[0], t[1] = -t[0], -t[1]
            selected_cameras[key] = {
                'K': K, 'distCoef': D, 'R': R, 't': t, 'M': M,
                'name': cam['name']
            }
        return selected_cameras
    
    def getRelativeCameraPose(self, ref_cam_name):
        '''
        Convert the absolute camera pose to relative camera pose.
        '''
        cam_ids = sorted(self.cam_params_dict)
        M10 = self.cam_params_dict[ref_cam_name]['M']
        M01 = b3dops.invertExtrinMat(M10)
        M2s_rel = []
        for cam_id in cam_ids:
            if cam_id == ref_cam_name: continue
            M20 = self.cam_params_dict[cam_id]['M']
            M21 = b3dops.transmitExtrinMats(M20, M01)
            M2s_rel.append(M21)
        return M2s_rel
        
    def gen2DJointsLabelFrom3DPanop(self, camera_list=None):
        '''
        Generate 2D pose from 3D label (Panoptic only has 3D label).
        '''
        pose3d_file_list = sorted(glob.glob(self.pose3d_file_dir + "/*"))
        if camera_list is None:
            camera_list = list(self.cam_params_dict.keys())

        if os.path.exists(self.pose2d_file_dir):
            shutil.rmtree(self.pose2d_file_dir)
        os.makedirs(self.pose2d_file_dir)

        for i, pose3d_file in enumerate(pose3d_file_list):
            frame_id = pose3d_file.split('/')[-1].split('_')[-1].split('.')[0]
            box_frame = json.load(open(pose3d_file))
            if len(box_frame['bodies']) == 0:
                os.remove(pose3d_file)
                continue
            if int(frame_id) % 100 == 0:
                print('Get 2D pose label for frame {} | camera {}.'.format(
                    frame_id, camera_list))

            for k, cam in self.cam_params_dict.items():
                if k not in camera_list:
                    continue

                person_pose2d_dict = {'joint_type': 'COCO19',
                                      'bodies': []}

                for body in box_frame['bodies']:
                    skel_3d = np.array(
                        body['joints19']).reshape((-1, 4)).transpose()
                    skel_2d = b3dops.projectPoints(
                        skel_3d[0:3,:], cam['K'], cam['distCoef'],
                        cam['R'], cam['t'])
                    confidence = np.expand_dims(skel_3d[3, :], 0)
                    skel_2d = np.concatenate((skel_2d, confidence), axis=0)
                    person_pose2d_dict['bodies'].append(
                        {'id': body['id'],'joints': skel_2d.tolist()})

                # save the 2D joints label
                save_path = os.path.join(self.pose2d_file_dir, cam['name'])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                pose2d_label_file = os.path.join(
                    save_path, str(frame_id).zfill(8) + '.json')
                json.dump(person_pose2d_dict, open(pose2d_label_file, 'w'))
    
    def fetchVideoFrameFile(self, camera_name, frame_id):
        return os.path.join(
            self.video_frame_dir, camera_name+'/{0:08d}.jpg'.format(frame_id))
    
    def getSingleFrameMultiView2DJoints(self, frame_id):
        '''
        Extract single frame multi-view joints.
        '''
        if self.pose2d_file_dir[-1] == '/':
            self.pose2d_file_dir = self.pose2d_file_dir[:-1]
        dataset_name = self.pose2d_file_dir.split('/')[-2]
        camera_folder_list = glob.glob(os.path.join(self.pose2d_file_dir,'*'))
        
        joints_dict = {}
        for camera_folder in camera_folder_list:
            camera_name = camera_folder.split('/')[-1]
            pose2d_json_file = os.path.join(
                self.pose2d_file_dir, camera_name,
                str(frame_id).zfill(8)+'.json')
            frame_joints = json.load(open(pose2d_json_file, 'r'))
            
            frame_joints_dict = {}
            for body in frame_joints['bodies']:
                person_id = body['id']
                joints_2d = np.array(body['joints'])[:, self.joints_ids]
                frame_joints_dict[(frame_id, person_id)] = joints_2d
                
            joints_dict[camera_name] = frame_joints_dict
        return joints_dict
    
    def getSingleFrameMultiViewBoxes(
            self, frame_id, box_joints_margin=1.2, box_ios_thold=0.7,
            box_size_thold=(20, 20), joints_inside_img_ratio=0.6,
            box_inside_img_ratio=0.6, resize=(128, 256), verbose=False,
            replace_old=False, img_postfix='.jpg'):
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
            for img_file in old_files:
                os.remove(img_file)
            reid_feat_file = os.path.join(save_crop_dir,'box_reid_feat.pkl')
            if os.path.exists(reid_feat_file):
                os.remove(reid_feat_file)
        
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
                if box[0] >= box[2] or box[1] >= box[3]: continue
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
            self, frame_id, num_prev_frames=0, trking_method='person_id',
            trk_feat_method='mean', reid_model=None, reid_log_file=None):
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
        if not os.path.exists(reid_feat_file):
            print('Re-ID feature file not found!')
            return None
        
        reid_feat = pkl.load(open(reid_feat_file, 'rb'))

        return reid_feat
            
    
    def genPtsCorrepFromBoxClus(
            self,boxfile_clusters,frame_id=None,noise_sz=None,verbose=False):
        '''
        Get 2D-2D point correspondences from box crop files clusters.
        '''
        if self.config is not None:
            correp_config = self.config['correspondence']
            noise_sz = correp_config['noise_sz']
            
        if verbose:
            print("\nGet 2D-2D correspondences:\n====================")
            
        # --- get parameters (num_cams, person_ids...)
        values = list(boxfile_clusters.values())
        while values[0] is None or len(values[0]) == 0: values.pop(0)
        frame_box_dir = os.path.join(
            self.boxcrop_dir, values[0][0].split('/')[-2])
        persons, cam_ids = set(), set()
        for person, boxfile_list in boxfile_clusters.items():
            persons.add(person)
            for boxfile in boxfile_list:
                panel_ids = boxfile.split('/')[-1].split('_')[0].split('-')
                cam_ids.add((int(panel_ids[0]), int(panel_ids[1])))
        cam_ids, persons = sorted(list(cam_ids)), sorted(list(persons))
            
        # --- get real number of people from GT if required
        if frame_id is not None:
            pose3d_gt_dict = json.load(open(os.path.join(
                self.pose3d_file_dir, str(frame_id).zfill(8) + '.json'), 'r'))
            persons_gt = []
            for body_3d in pose3d_gt_dict['bodies']:
                persons_gt.append(body_3d['id'])
            persons = sorted(persons_gt)

        # Step 2: Establish 2D-2D point correpondences
        box_joints_map = pkl.load(
            open(os.path.join(frame_box_dir, 'box_joints_map.pkl'), 'rb'))
        
        corresp_dict = {}
        for cam_id in cam_ids:
            corresp_dict[cam_id] = {'keypoints': [], 'box_files': []}

        for person in persons:
            if verbose:
                print('Get pt correspondences of person {}.'.format(person))
            
            # set invisible people to np.nan, if visible, will change later
            for cam_id in cam_ids:
                corresp_dict[cam_id]['keypoints'].append(
                    np.ones((len(self.joints_ids), 2)) * np.nan)
                corresp_dict[cam_id]['box_files'].append(None)
            
            if person not in boxfile_clusters:
                continue

            for boxfile in boxfile_clusters[person]:
                panel_ids = boxfile.split('/')[-1].split('_')[0].split('-')
                cam_id = (int(panel_ids[0]), int(panel_ids[1]))
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
        save_file = os.path.join(frame_box_dir, 'box_gen_corresp_dict.pkl')
        pkl.dump(corresp_dict, open(save_file, 'wb'))
        n_persons = len(persons)
        if verbose:
            print('\nSaved to:\n\"{}\".'.format(save_file))
            print('====================\n')
        return corresp_dict, n_persons
    
    def getFrameEst3DPose(self, frame_id, ref_cam_id):
        '''Get estimated 3D human pose by frame ID.
        '''
        pose3d_est_file = os.path.join(
            self.boxcrop_dir,'frame'+str(frame_id).zfill(8),'pose_est.json')
        pose3d_est = json.load(open(pose3d_est_file, 'r'))
        pose3d_human = np.array(pose3d_est['Pts'])
        pose3d_camera = np.array(pose3d_est['Ms'])
        
        if not (ref_cam_id == pose3d_est['world_camera_id']):
            M2 = pose3d_camera[ref_cam_id]
            R, t = M2[:, :3], M2[:, 3]
            pose3d_human = R.dot(pose3d_human.T).T
        return pose3d_human
    
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
        Pts_w = Pts_w[:, [2, 0, 1]]  # Maybe because the world coordinate
        # Pts_w[:, 2] = -Pts_w[:, 2]   # of Panoptic is not the ground plane 
        Pts_w = -Pts_w
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
            joints3d_gt = np.array(body_3d['joints19']).reshape((-1, 4))[:,:3]
            joints3d_gt = joints3d_gt[self.joints_ids]
            Pts_by_person[int(body_3d['id'])] = {'pose3d_GT': joints3d_gt}

        # --- Step 3: convert estimated relative 3D pose to absolut 3D pose
        assert np.array(Pts_est_rela).shape[1] == 3
        Pts_est_abs = np.nan * np.ones_like(Pts_est_rela)
        idx = np.arange(len(Pts_est_rela))[~np.isnan(Pts_est_rela[:, 0])]
        Pts_est_abs[idx] = np.linalg.pinv(R).dot((Pts_est_rela[idx] - t).T).T

        # --- Step 4: solve scale ambiguity, align GT and Est
        n_joints = len(self.joints_ids)
        Pts_gt, Pts_est = [], []
        for k, person_id in enumerate(sorted(list(Pts_by_person.keys()))):
            Jnts_gt = Pts_by_person[person_id]['pose3d_GT']
            Jnts_est = Pts_est_abs[k*n_joints:(k+1)*n_joints]

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

        # --- Last step: rotate ground plane to upwards, this is only
        # needed for Panoptic dataset for visualization purpose
        Pts_gt, Pts_est = Pts_gt[:, [2, 0, 1]], Pts_est[:, [2, 0, 1]]
        Pts_gt, Pts_est = -Pts_gt, -Pts_est
        Pts_est[:, 0], Pts_est[:, 1] = -Pts_est[:, 0], -Pts_est[:, 1]

        return Pts_gt, Pts_est, Pts_by_person, scale
    