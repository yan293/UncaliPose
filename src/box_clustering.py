"""
   This scipt defines functions for bounding boxes clustering and evaluation.
   Author: Yan Xu
   Date: Jul 02, 2022
"""
import os, sys
CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(CURRENT_DIR)
from misc import getFilesOfType
from clustering.clustering_wrapper import perform1DClustering, evalMultiClustMethods

from sklearn.cluster import KMeans, kmeans_plusplus,\
    SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from collections import defaultdict
from scipy.spatial import distance
from sklearn import preprocessing
from statistics import mode
import pickle as pkl
import numpy as np
import time
import copy


def prepareClusteringData(box_feat_dict):
    '''
    Get variables, including, feature, file names, mustlinks, cannotlinks.
    '''
    cam_prsn_ids, person_ids, boxfeats, boxfiles = [], [], [], []
    for boxfile in sorted(box_feat_dict.keys()):
        boxfiles.append(boxfile)
        boxfeat = box_feat_dict[boxfile]
        cam_id = boxfile.split('/')[-1].split('_')[0]
        prsn_id = boxfile.split('/')[-1].split('_')[1]
        cam_prsn_ids.append((cam_id, prsn_id))
        person_ids.append(int(prsn_id))
        boxfeats.append(boxfeat)
    boxfeats = np.array(boxfeats)
    person_ids = np.array(person_ids)

    mustlinks, cannotlinks = [], []
    for i in range(len(boxfeats) - 1):
        cam_prsn_id_i = cam_prsn_ids[i]
        for j in range(i+1, len(boxfeats)):
            cam_prsn_id_j = cam_prsn_ids[j]
            if cam_prsn_id_i[0] == cam_prsn_id_j[0] \
               and cam_prsn_id_i[1] != cam_prsn_id_j[1]:
                cannotlinks.append((i, j))
    return boxfeats, boxfiles, person_ids, mustlinks, cannotlinks

    
def getBoxfileClusters(cluster_labels, boxfiles, gt_person_id=True):
    '''
    Get the box file clusters from the cluster labels.
    '''
    boxfile_clusters = defaultdict(list)
    for i in range(len(cluster_labels)):
        key = cluster_labels[i]
        boxfile_clusters[key].append(boxfiles[i])
    if gt_person_id is False:
        return boxfile_clusters
        
    # --- if the person id is provided, use that
    boxfile_clusters_ = {}
    for _, boxfile_list in boxfile_clusters.items():
        person_id_list = []
        for boxfile in boxfile_list:
            person_id_list.append(boxfile.split('/')[-1].split('_')[1])
        person_id = int(mode(person_id_list))
        boxfile_clusters_[person_id] = boxfile_list
    return boxfile_clusters_


def rmvBoxesFromSameView(boxfile_clusters):
    '''
    If several people viewed from the same camera are clustered together,
    remove all of them (the confusing ones) from the cluster.
    '''
    boxfile_clusters_ = copy.deepcopy(boxfile_clusters)
    for person_id, boxcrop_file_list in boxfile_clusters_.items():
        people_viewed_by_cam = defaultdict(list)
        
        for boxcrop_file in boxcrop_file_list:
            ids = boxcrop_file.split('/')[-1].split('.')[0].split('_')
            cam_name = ids[0]
            people_viewed_by_cam[cam_name].append(boxcrop_file)
        
        for cam_name, people in people_viewed_by_cam.items():
            if len(people) > 1:  # normally, should be 1
                for person in people:
                    boxcrop_file_list.remove(person)
        boxfile_clusters_[person_id] = boxcrop_file_list
    return boxfile_clusters_


def rmvZeroAndOneClusters(boxfile_clusters):
    '''
    If the size of a cluster is 0 or 1, it is useless for obtaining point
    correspondences.  We thus remove such clusters
    '''
    rm_keys = []
    for key, val in boxfile_clusters.items():
        if len(val) < 2:
            rm_keys.append(key)
    
    boxfile_clusters_ = copy.deepcopy(boxfile_clusters)
    for key in rm_keys:
        box_file_clusters_[key] = []
    if not bool(box_file_clusters_):
        print("Clustering failed to output useful correspondences!")
        box_file_clusters_ = None
    return box_file_clusters_


def multiViewHumanBoxesClustering(
        box_feat_dict, n_persons=None, method='kmeans_ssc', dist='cosine',
        size_min=2, size_max=None, n_init=100, eval_clust=True, verbose=True,
        gt_person_id=True,compare_clust_methods=False,save=False,config=None):
    '''
    Cluster multi-view human bounding boxes using ReID feature.
    '''
    if config is not None:
        compare_clust_methods = config['compare_clust_methods']
        verbose = config['verbose']
        method = config['method']
        n_init = config['n_init']
    
    boxfeats, boxfiles, person_ids, mustlinks, cannotlinks = \
        prepareClusteringData(box_feat_dict)
    
    if n_persons is None:
        n_persons = len(set(person_ids))
    
    if compare_clust_methods:
        clustering_methods = [
            'kmeans', 'kmeans++', 'kmeans_ssc', 'spectral', 'gmm', 'dbscan',
            'agglomerative', 'kmeans_constrained', 'kmeans_ssc']
        distances = ['cosine', 'euclidean']
    else:
        clustering_methods = [method]
        distances = [dist]
    
    clustering_result = evalMultiClustMethods(
        boxfeats, n_persons, methods=clustering_methods, dists=distances,
        n_init=n_init, size_min=size_min, size_max=size_max, verbose=verbose,
        mustlinks=mustlinks, cannotlinks=cannotlinks, label_true=person_ids)
    cluster_label = clustering_result[method][dist]['cluster_labels']
    
    boxfile_clusters = getBoxfileClusters(
        cluster_label, boxfiles, gt_person_id=gt_person_id)
    boxfile_clusters_1 = rmvBoxesFromSameView(boxfile_clusters)
    boxfile_clusters_2 = rmvBoxesFromSameView(boxfile_clusters_1)
    
    return boxfile_clusters_2
