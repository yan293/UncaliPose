# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os, sys
MAIN_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.append(MAIN_DIR)
sys.path.append('.')

import argparse
from os import mkdir

import torch
from torch.backends import cudnn
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger

from data.transforms import build_transforms
from data.datasets import read_image
import pickle as pkl
import numpy as np
import glob
import os
import time
import pdb


def preprocess_img(img_dir, transform):

    img_types = ['*.jpg', '*.png']
    img_file_list = []
    print(img_dir)
    for img_type in img_types:
        img_file_list.extend(glob.glob(os.path.join(img_dir, img_type)))
        
    imgs = []
    for img_file in img_file_list:
        img = read_image(img_file)
        img_transform = transform(img)
        imgs.append(img_transform)
    imgs = torch.stack(imgs, dim=0)

    return img_file_list, imgs


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    
    # ======================= FOR WILDPOSE ======================= #
    model.eval()
    
    feats = {}
    def get_features(name):
        def hook(model, input, output):
            feats[name] = output.detach()
        return hook
    
    # the feature after batch norm is the same as the model output
    model.bottleneck.register_forward_hook(get_features('f_i'))
    model.base.layer4[2].conv3.register_forward_hook(get_features('f_conv'))
    
    test_transforms = build_transforms(cfg, is_train=False)
    img_file_list, imgs = preprocess_img(cfg.TEST.IMG_DIR, test_transforms)
    
    output = model(imgs)
    reid_feat = output.detach().numpy()
    
    img_to_reid_feat_map = {}
    for i in range(len(img_file_list)):
        img_to_reid_feat_map[img_file_list[i]] = reid_feat[i]
        
    feat_save_file = os.path.join(cfg.TEST.IMG_DIR, 'box_reid_feat.pkl')
    pkl.dump(img_to_reid_feat_map, open(feat_save_file, 'wb'))

    # =========================== END =========================== #


if __name__ == '__main__':
    main()
