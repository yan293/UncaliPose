"""
   This scipt defines functions for COP (source-constrained optimization)
   post processing.
   Author: Yan Xu
   Update: May 04, 2022
"""

import os, sys
CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(CURRENT_DIR)

from cop_kmeans import transitive_closure, violate_constraints

from k_means_constrained import KMeansConstrained
from collections import defaultdict
from sklearn import preprocessing
from scipy.spatial import distance
import numpy as np
import copy


def sscClustering(
        X, n_clusters, size_min=2, size_max=None, dist='cosine',
        mustlinks=[], cannotlinks=[], n_init=100, verbose=True):
    '''
    Size and source constrained clustering.
    '''
    if n_clusters == 1:
        cluster_label_ssc = np.array([0] * len(X))
    else:
        # --- data normalization
        if dist == 'euclidean':
            X1 = copy.deepcopy(X)
        elif dist == 'cosine':
            X1 = preprocessing.normalize(copy.deepcopy(X), axis=1)

        # --- step 1: size-constrained pre-clustering
        if size_max is None:
            size_max = len(X1) - size_min * n_clusters + 1
        model = KMeansConstrained(
            n_clusters=n_clusters, size_min=size_min, size_max=size_max,
            init='k-means++', random_state=n_init).fit(X1)
        cluster_label = model.labels_

        # --- step 2: source-constrained post-clustering
        cluster_label_ssc = sourceConstrainedPostClustering(
            X1, cluster_label, mustlinks=mustlinks, cannotlinks=cannotlinks,
            min_size=size_min, max_size=size_max)
        
    return cluster_label_ssc


def sourceConstrainedPostClustering(
        data_feats, cluster_labels, mustlinks=[], cannotlinks=[],
        min_size=None, max_size=None, dist='cosine'):
    '''
    Perform source-constrained clustering on top of the result of
    size-constrained clustering.
    '''
    data_feats_copy = copy.deepcopy(data_feats)
    if dist == 'cosine':
        data_feats_copy = preprocessing.normalize(data_feats_copy, axis=1)
        
    # --- step 1: find hard samples that violates 'cannotlink'
    ml_graph, cl_graph = transitive_closure(
        mustlinks, cannotlinks, len(data_feats))
    hard_samples, hard_groups, assigned_clusters = findHardSamples(
        cluster_labels, ml_graph, cl_graph)
    
    # --- if there are empty clusters, don't do anything (temporarily)
    for key, val in assigned_clusters.items():
        if len(val) == 0:
            return cluster_labels
    
    # --- step 2: Assign the hard group to clusters
    while len(hard_groups) > 0:
        group, avail_cluster_ids, dist_mat = findNextGroupToAssign(
            data_feats_copy, hard_groups, assigned_clusters, max_size,
            dist_metric='euclidean', cannotlink_graph=cl_graph)
        assigned_clusters = assignCannotLinkGroup(
            group, avail_cluster_ids, dist_mat, assigned_clusters)
        hard_groups.remove(group)
    
    # --- step 3: update cluster centers and labels
    cluster_labels_cop = cluster_labels.copy()
    for cluster_id, members in assigned_clusters.items():
        for element in members:
            cluster_labels_cop[element] = cluster_id
    
    return cluster_labels_cop


def findHardSamples(cluster_labels, mustlink_graph, cannotlink_graph):
    '''
    Find hard samples that viaolate cannot link constraints.
    '''
    # --- step 1: find hard samples
    hard_samples, correct_clusters = [], {}
    for i, cluster_label in enumerate(cluster_labels):
        correct_clusters[cluster_label] = []
        violate = violate_constraints(
           i, cluster_label, cluster_labels, mustlink_graph, cannotlink_graph)
        if violate:
            hard_samples.append(i)
    
    for i, cluster_label in enumerate(cluster_labels):
        if i not in hard_samples:
            correct_clusters[cluster_label].append(i)
            
    # --- divide hard samples into cannot link groups
    hard_groups = []
    hard_samples_copy = copy.deepcopy(hard_samples)
    while len(hard_samples_copy) != 0:
        data_i = hard_samples_copy.pop(0)
        pair = [data_i]
        for data_j in cannotlink_graph[data_i]:
            if data_j in hard_samples_copy:
                pair.append(data_j)
                hard_samples_copy.remove(data_j)
        hard_groups.append(pair)
    return hard_samples, hard_groups, dict(correct_clusters)


def findNextGroupToAssign(
        data_feats, hard_groups, clusters, max_size,
        dist_metric='euclidean', cannotlink_graph=None):
    '''
    Sort the hard pairs.  List the pair whose one element has the smallest
    distance to one of the cluster centers.
    '''
    dist_ratio_max = -np.inf
    for hard_group in hard_groups:
        
        notfull_clusters = findNotFullClusters(clusters, max_size)
        dist_mat, cluster_ids = getDistanceMatrix(
            data_feats, hard_group, notfull_clusters, dist_metric=dist_metric,
            cannotlink_graph=cannotlink_graph)
        dist_ratio = getMatDistRatio(dist_mat)
        
        if dist_ratio == np.inf:
            return hard_group, cluster_ids, dist_mat
        
        if dist_ratio > dist_ratio_max:
            group = hard_group
            avail_cluster_ids = cluster_ids
            dist_mat_of_group = dist_mat
            dist_ratio_max = dist_ratio
    
    return group, avail_cluster_ids, dist_mat_of_group


def findNotFullClusters(clusters, max_size):
    '''
    Find clusters that has not reach the max size.
    '''
    notfull_clusters = {}
    for cluster_id, members in clusters.items():
        if len(members) < max_size:
            notfull_clusters[cluster_id] = members
    return notfull_clusters


def getDistanceMatrix(data_feats, data_ids, clusters,
        dist_metric='euclidean', cannotlink_graph=None):
    '''
    '''
    cluster_ids, cluster_centers = [], []
    for cluster_id, members in clusters.items():
        cluster_ids.append(cluster_id)
        assert len(members) != 0
        cluster_centers.append(
            np.mean(data_feats[np.array(members)], axis=0))
        
    dist_mat = distance.cdist(
        data_feats[np.array(data_ids)], cluster_centers,
        metric=dist_metric)
    
    if cannotlink_graph is not None:
        for i, data_id in enumerate(data_ids):
            for j, cluster_id in enumerate(cluster_ids):
                for cannotlink_data in cannotlink_graph[data_id]:
                    if cannotlink_data in clusters[cluster_id]:
                        dist_mat[i][j] = np.nan
    return dist_mat, cluster_ids


def getMatDistRatio(dist_mat):
    '''
    Get the distance ratio of a distance matrice.  The distance ratio of
    each row is defined as: the second smallest element / the smallest 
    element.  The distance ratio of the matrice is the largest value of
    the distance ratios of all the rows.
    '''
    dist_ratio_mat = -np.inf
    for row in dist_mat:
        notnan_sort = np.sort(row[~np.isnan(row)])
        if len(notnan_sort) == 1:
            return np.inf
        dist_ratio_row = notnan_sort[1] / notnan_sort[0]
        dist_ratio_mat = max(dist_ratio_mat, dist_ratio_row)
    return dist_ratio_mat


def assignCannotLinkGroup(
        data_ids, cluster_ids, dist_mat, assigned_clusters):
    '''
    Assigin a group of hard samples to different clusters.
    '''
    data_cluster_zip = []
    for data_id in data_ids:
        temp = []
        for cluster_id in cluster_ids:
            temp.append((data_id, cluster_id))
        data_cluster_zip.append(temp)

    assigned_clusters_copy = assigned_clusters.copy()
    while len(dist_mat) > 0:
        r, c = findBestElementToAssign(dist_mat)
        data_id, cluster_id = data_cluster_zip[r][c]
        assigned_clusters_copy[cluster_id].append(data_id)

        dist_mat = np.delete(dist_mat, r, axis=0)
        dist_mat = np.delete(dist_mat, c, axis=1)
        del data_cluster_zip[r]
        for row in data_cluster_zip:
            del row[c]
    return assigned_clusters_copy


def findBestElementToAssign(dist_mat):
    '''
    Find the first and second smallest elements in each row.  If the ratio
    second/first is the largest among all rows, return the location of the
    (row, col) smallest element of that row.
    '''
    row_dist_ratio = -np.Inf
    r, c = 0, 0
    for i, row in enumerate(dist_mat):
        if np.count_nonzero(~np.isnan(row)) == 1:
            return i, np.argwhere(~np.isnan(row))[0][0]
        
        not_nans = sorted(row[~np.isnan(row)])
        if row_dist_ratio < float(not_nans[1]) / not_nans[0]:
            row_dist_ratio = float(not_nans[1]) / not_nans[0]
            r = i
            c = np.where(row==not_nans[0])[0][0]
    return r, c


# def assignEmptyClusters(hard_groups, clusters, max_size=None):
#     '''
#     If there are empty clusters, pick one element from the largest hard groups
#     and assign the element to an empty cluster, until no empty clusters left.
#     '''
#     def findLargestGroup(groups):
#         group_size = 0
#         for i, group in enumerate(groups):
#             if len(group) > group_size:
#                 largest_group, group_id, group_size = group, i, len(group)
#         return largest_group, group_id
    
#     hard_groups_1 = hard_groups.copy()
#     for cluster_id, members in clusters.items():
#         if len(members) == 0:
#             largest_group, group_id = findLargestGroup(hard_groups_1)
            
#         # 有空的cluster means，有一个人完全看不见
#         # 有大的hard group means，有一个camera容易引起混淆
#         # 这个camera里的人有一个，属于看不见的那个cluster，怎么找到这个人？
#         # 或者把这个camera，完全移除会影响吗，反正目标是human pose
#         # 但是有空的cluster却不行，说明人没有完全重建
#     return None
