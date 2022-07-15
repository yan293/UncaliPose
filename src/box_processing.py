"""
   This scipt defines functions for processing bounding boxes and joints.
   Format: (x_tl, y_tl, x_br, y_br, joints_in_img_ratio, box_in_img_ratio).
   Author: Yan Xu
   Update: Nov 29, 2021
"""
import os, sys
CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(CURRENT_DIR)

from collections import defaultdict
import matplotlib.pyplot as plt
from misc import getFilesOfType
import numpy as np
import copy
import time
import cv2
import os


def countNumJointsInsideImage(im_size, joints):
    '''
    Count number of joints inside image.
    
    Input:
        im_size: A tuple, (image.shape[0], image.shape[1]).
        joints: A [2 x N] array, the first two rows are joint
                coordinates.
    Output:
        count: number of joints inside image.
        ratio: ratio of number of joints inside image.
    '''
    im_H, im_W = im_size[0], im_size[1]
    joints = np.array(joints)
    joints_vis = np.ones_like(np.array(joints)) * np.nan
    total = joints.shape[1]
    count = 0
    for i in range(len(joints[0])):
        x_inside = ~np.isnan(joints[1][i]) and 0 <= joints[1][i] <= im_H
        y_inside = ~np.isnan(joints[0][i]) and 0 <= joints[0][i] <= im_W
        if x_inside and y_inside:
            count += 1
            joints_vis[:, i] = joints[:, i]
    return count, float(count) / float(total), joints_vis


def cutBoxAroundJoints(im_size, joints, margin_ratio=1.1):
    '''
    Generate a bounding box around the given joints and image size.
    
    Input:
        im_size: A tuple, (image.shape[0], image.shape[1]).
        joints: A [2 x N] array, the first two rows are joint
                coordinates.
        margin_ratio: Box to joints ratio (>= 1).
    Output:
        Return (x_tl, y_tl, x_br, y_br,
                joints_in_img_ratio, box_in_img_ratio).
    '''
    # image size
    im_H, im_W = im_size[0], im_size[1]
    
    # joints inside image
    _, joints_in_img_ratio, _ = countNumJointsInsideImage(im_size, joints)
    
    # joints boundary
    x_min, x_max = np.nanmin(joints[1]), np.nanmax(joints[1])
    y_min, y_max = np.nanmin(joints[0]), np.nanmax(joints[0])
    
    # consider margin (box > joints)
    if isinstance(margin_ratio, float):
        margin_ratio_x, margin_ratio_y = margin_ratio, margin_ratio
    elif len(margin_ratio) == 2:
        margin_ratio_x, margin_ratio_y = margin_ratio[1], margin_ratio[0]
    dx = (margin_ratio_x - 1.) * (x_max - x_min) / 2.
    dy = (margin_ratio_y - 1.) * (y_max - y_min) / 2.
    x_min, x_max = x_min - dx, x_max + dx
    y_min, y_max = y_min - dy, y_max + dy
    
    # check if the box is out the image range
    x_tl = int(max(0, x_min))
    y_tl = int(max(0, y_min))
    x_br = int(min(im_H, x_max))
    y_br = int(min(im_W, y_max))
    
    # the ratio of box inside the image
    area = (x_max - x_min) * (y_max - y_min)
    area_in = (x_br - x_tl) * (y_br - y_tl)
    box_in_img_ratio = float(area_in) / area
    return (x_tl, y_tl, x_br, y_br, joints_in_img_ratio, box_in_img_ratio)


def intersectionOverSelf(box1, box2):
    '''
    Compute the ratio of intersection to self area of two boxes.
    
    Input:
        box1: [x_tl, y_tl, x_br, y_br]
        box2: [x_tl, y_tl, x_br, y_br]
    Output:
        ios_box1: Intersection of Self (IOS) of box1 [0, 1]
        ios_box2: Intersection of Self (IOS) of box2 [0, 1]
    '''
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    ios_box1 = float(intersection) / box1_area
    ios_box2 = float(intersection) / box2_area
    return ios_box1, ios_box2
        

def removeBlockedBoxes(boxes, box_ios_thold=0.7):
    '''
    Remove boxes that have large IoS with other boxes. Since human have
    similar sizes, a smaller box means person is further from the camera.
    When the IoS is large, the person is likely to be blocked.
    
    Input:
        boxes: A list of boxes, [box_1, box_2, ...], where 'box_i': [x_tl,
            y_tl, x_br, y_br, **], ** are other box infomation (useless here).
        box_ios_thold: IoS threshold, above which will be removed.
    Output:
        boxes_: Selected boxes, [[x_tl, y_tl, x_br, y_br, **], ...].
    '''
    if len(boxes) < 2: return boxes, range(len(boxes))
    
    n_box = len(boxes)
    boxes_ = [tuple(list(box) + [idx]) for idx, box in enumerate(boxes)]
    
    # print(np.array(boxes))
    
    # compute IoS
    box_ios = {}
    for box in boxes_: box_ios[box] = [0.0, 'front']
    for i in range(n_box - 1):
        for j in range(i + 1, n_box):
            ios_i, ios_j = intersectionOverSelf(boxes[i], boxes[j])
            box_ios[boxes_[i]][0] = max(ios_i, box_ios[boxes_[i]][0])
            box_ios[boxes_[j]][0] = max(ios_j, box_ios[boxes_[j]][0])
            
            box_len_i = abs(boxes_[i][2] - boxes_[i][0])
            box_len_j = abs(boxes_[j][2] - boxes_[j][0])
            loc_i, loc_j = box_ios[boxes_[i]][1], box_ios[boxes_[j]][1]
            if loc_j == 'front' and ios_j > 0. and box_len_j < box_len_i:
                loc_j = 'back'
            if loc_i == 'front' and ios_i > 0. and box_len_i < box_len_j:
                loc_i = 'back'
            box_ios[boxes_[i]] = [max(ios_i, box_ios[boxes_[i]][0]), loc_i]
            box_ios[boxes_[j]] = [max(ios_j, box_ios[boxes_[j]][0]), loc_j]
    
    # select small IoS ones (less likely blocked)
    boxes_select, idxes = [], []
    for box, val in box_ios.items():
        ios, loc = val[0], val[1]
        if loc == 'back' and ios > box_ios_thold:
            continue
        boxes_select.append(box[:-1])
        idxes.append(int(box[-1]))
    return boxes_select, idxes
    
    
def removeSmallBoxes(boxes, box_size_thold=(20, 20)):
    '''
    Remove small boxes.
    
    Input:
        boxes: A list of boxes, [box_1, box_2, ...], where 'box_i': [x_tl,
            y_tl, x_br, y_br, **], ** are other box infomation (useless).
        box_size_thold: box size threshold, separately for x and y.
    Output:
        boxes_: Selected boxes, [[x_tl, y_tl, x_br, y_br, **], ...].
    '''
    boxes_, idxes = [], []
    for i, box in enumerate(boxes):
        if box[2] - box[0] > box_size_thold[0] and \
           box[3] - box[1] > box_size_thold[1]:
            boxes_.append(box)
            idxes.append(i)
    return boxes_, idxes
    
    
def removeOutsideViewJoints(boxes, joints_inside_img_ratio=0.6):
    '''
    Remove boxes whose #joints inside image ratio smaller than a threshold.
    
    Input:
        boxes: A list of boxes, [box_1, box_2, ...], where 'box_i': [x_tl,
            y_tl, x_br, y_br, inside_img_ratio].
        joints_inside_img_ratio: box size threshold, separately for x and y.
    Output:
        boxes_: Selected boxes, [[x_tl, y_tl, x_br, y_br, **], ...].
    '''
    boxes_, idxes = [], []
    for i, box in enumerate(boxes):
        assert len(box) > 4
        if box[4] > joints_inside_img_ratio:
            boxes_.append(box)
            idxes.append(i)
    return boxes_, idxes
    
    
def removeOutsideViewBoxes(boxes, box_inside_img_ratio=0.6):
    '''
    Remove boxes whose inside image ratio are smaller than a threshold.
    
    Input:
        boxes: A list of boxes, [box_1, box_2, ...], where 'box_i': [x_tl,
            y_tl, x_br, y_br, inside_img_ratio].
        box_inside_img_ratio: box size threshold, separately for x and y.
    Output:
        boxes_: Selected boxes, [[x_tl, y_tl, x_br, y_br, **], ...].
    '''
    boxes_, idxes = [], []
    for i, box in enumerate(boxes):
        assert len(box) > 5
        if box[5] > box_inside_img_ratio:
            boxes_.append(box)
            idxes.append(i)
    return boxes_, idxes


def cropBoxesInImage(image_file, boxes, save_dir=None,
                     prefixes=None, img_postfix='.jpg', resize=None):
    '''
    Crop the boxes in an image, save crops if "save_dir" is not "None".
    
    Input:
        image_file: image file.
        boxes: A list of boxes, [box_1, box_2, ...], where 'box_i': [x_tl,
            y_tl, x_br, y_br, **], ** are other box infomation (useless).
        save_dir: the save directory, not save if "None".
        prefixes: the prefixes of box crop file, [prefix_1, prefix_2, ...].
    Output:
        box_crops: Box crops, [[HxWx3], ...].
    '''
    im = cv2.imread(image_file)
    im_size = im.shape[0], im.shape[1]
    
    box_crops, save_files = [], []
    for i, box in enumerate(boxes):
        x_tl, y_tl, x_br, y_br = box[0], box[1], box[2], box[3]
        box_crop = im[int(x_tl):int(x_br), int(y_tl):int(y_br)]
        if resize is not None:
            box_crop = cv2.resize(box_crop, resize)
        box_crops.append(box_crop)
        
        # save the box crop
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            box_crop_name = str(x_tl) + '_' + str(y_tl) + '_' + \
                            str(x_br) + '_' + str(y_br) + img_postfix
            if prefixes is not None:
                assert len(prefixes) == len(boxes)
                box_crop_name = prefixes[i] + '_' + box_crop_name
            save_file = os.path.join(save_dir, box_crop_name)
            cv2.imwrite(save_file, box_crop)
            save_files.append(save_file)
    return box_crops, save_files


def extractBoxReIDFeature(boxcrop_dir, reid_model=None, log_file=None):
    '''
    Extract re-ID feature of the bounding boxes.
    '''
    import os, sys
    CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    sys.path.append(CURRENT_DIR)
    import subprocess
    
    box_file_list = getFilesOfType(boxcrop_dir, type_list=['*.jpg', '*.png'])
    if len(box_file_list) == 0:
        return None
    print('\nExtract re-ID feature\n=============\nBoxes under {}.'.format(
        boxcrop_dir))
    
    if reid_model is None:
        reid_model = 'src/reid_strong_baseline/FOR_SSC/pretrained_models/'\
            'market_resnet50_model_120_rank1_945.pth'
    if log_file is not None:
        stdout = open(log_file, 'w')
    else:
        stdout = subprocess.PIPE
    
    cmdline_list = [
        'python3', 'src/reid_strong_baseline/tools/extract_fi_feat_ssc.py',
        '--config_file=src/reid_strong_baseline/configs/'\
            'softmax_triplet_with_center_ssc.yml',
        'TEST.WEIGHT', reid_model,
        'TEST.IMG_DIR', boxcrop_dir]
    
    time_1 = time.time()
    process = subprocess.Popen(
        cmdline_list, stdout=stdout, universal_newlines=True)

    if log_file is None:
        while True:
            output = process.stdout.readline()
            # print(output.strip())
            return_code = process.poll()
            if return_code is not None:
                # print('RETURN CODE', return_code)
                for output in process.stdout.readlines():
                    print(output.strip())
                break
    print('Re-ID feature saved to {}.'.format(
        os.path.join(boxcrop_dir, 'box_reidfeat_dict.pkl')))
    print('[{:.2f} seconds]\n=============\n'.format(time.time() - time_1))
    return None
