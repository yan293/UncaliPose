"""
   This scipt defines functions for clustering using different methods.
   Author: Yan Xu, CMU
   Update: Jul 04, 2022
"""
import os, sys
CURRENT_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(CURRENT_DIR)

from clustering_metric import getClustMetrics
from ssc_clustering import sscClustering
from k_means_constrained import KMeansConstrained  # conda install
from sklearn.mixture import GaussianMixture
import sklearn.cluster as skclust
from sklearn import preprocessing
from statistics import mode
import numpy as np
import copy
import time

    
def perform1DClustering(
        X, n_clusters, method='kmeans_ssc', dist='cosine', n_init=100,
        mustlinks=[], cannotlinks=[], size_min=2, size_max=None,verbose=True):
    '''
    Perform 1D feature clustering.
    '''
    if n_clusters == 1:
        return {'cluster_labels': np.array([0] * len(X))}
    
    # --- feature normalization
    if dist == 'euclidean':
        X_ = copy.deepcopy(X)
    elif dist == 'cosine':
        X_ = preprocessing.normalize(copy.deepcopy(X), axis=1)
    
    # --- clustering using different methods
    time_start = time.time()
    if method == 'kmeans_ssc':
        cluster_label = sscClustering(
            X, n_clusters, size_min=size_min, size_max=size_max,
            mustlinks=mustlinks, cannotlinks=cannotlinks, dist=dist,
            n_init=n_init, verbose=verbose)
    elif method in ['kmeans', 'kmeans++', 'spectral', 'agglomerative',\
                    'dbscan', 'gmm', 'kmeans_constrained']:
        if method == 'kmeans':
            model = skclust.KMeans(
                n_clusters=n_clusters,init='random', n_init=n_init).fit(X_)
        elif method == 'kmeans++':
            model = skclust.KMeans(
                n_clusters=n_clusters, init='k-means++', n_init=n_init).fit(X_)
        elif method == 'spectral':
            model = skclust.SpectralClustering(
                n_clusters=n_clusters, n_init=n_init, assign_labels='discretize',
                random_state=0).fit(X_)
        elif method == 'agglomerative':
            model = skclust.AgglomerativeClustering(n_clusters=n_clusters).fit(X_)
        elif method == 'dbscan':
            model = skclust.DBSCAN(eps=0.4, min_samples=2,metric='cosine').fit(X_)
        elif method == 'gmm':
            model = GaussianMixture(n_components=n_clusters).fit(X_)
            model.labels_ = model.predict(X_)
        elif method == 'kmeans_constrained':
            if size_max is None: size_max = len(X) - size_min * n_clusters + 1
            model = KMeansConstrained(
                n_clusters=n_clusters, size_min=size_min, size_max=size_max,
                init='k-means++', random_state=n_init).fit(X_)
        cluster_label = model.labels_
        
    if verbose:
        print('[{:.2f} seconds] \"{}\" | \"{}\" | {} clusters.'.format(
            time.time() - time_start, method, dist, n_clusters))
        
    return {'cluster_labels': cluster_label}


def evalMultiClustMethods(
        X, n_clusters, methods=['kmeans_ssc'], dists=['cosine', 'euclidean'],
        n_init=100, size_min=2, size_max=None, label_true=None, verbose=True,
        mustlinks=[], cannotlinks=[]):
    '''
    Compare different clustering methods, and distances as well.
    '''
    assert len(methods) != 0
    if verbose:
        print('\nPerform clustering:\n=============')
    clustering_result, metrics = {}, {}
    for method in methods:
        clustering_result[method] = {}
        metrics[method] = {}
        for dist in dists:
            method_result = perform1DClustering(
                X, n_clusters, method=method, dist=dist, n_init=n_init,
                size_min=size_min, size_max=size_max, mustlinks=mustlinks,
                cannotlinks=cannotlinks, verbose=verbose)
            
            label_pred = method_result['cluster_labels']
            clustering_result[method][dist] = {'cluster_labels': label_pred}
            
            if label_true is not None:
                metrics[method][dist] = getClustMetrics(label_true,label_pred)
    if verbose:
        if label_true is not None:
            printClusterMetricsTable(metrics)
        print('\n=============\n')

    return clustering_result


def printClusterMetricsTable(metric_dict):
    '''
    Print clustering metrics in table.
    
    Input:
        metric_dict: {'method': {'distance': {}}}
    '''
    method_list = sorted(metric_dict.keys())
    distance_list = sorted(metric_dict[method_list[0]].keys())
    metric_list = list(metric_dict[method_list[0]][distance_list[0]].keys())
    
    name_len = 0
    for mthd in method_list:
        name_len = max(name_len, len(mthd))

    tab_title = '\n{:<'+str(name_len + 5)+'}{:<15}'+'{:<10}'*len(metric_list)
    print(tab_title.format(*list(['Method','Distance']+metric_list)),end ='')
    
    for mthd in method_list:
        for dist in distance_list:
            prefix = '\n{:<'+str(name_len + 5)+'}{:<15}'
            print(prefix.format(mthd, dist), end='')
            for met in metric_list:
                print('{:<10.3f}'.format(metric_dict[mthd][dist][met]),end='')


def elbowSearch(
        X, n_min=None, n_max=None, size_min=2, size_max=None,
        method='kmeans++', dist='cosine', n_init=100):
    '''
    Elbow search to find the best guess of the number of clusters.
    '''
    if n_min is None: n_min = 2
    if n_max is None: n_max = len(X) - 1
    if dist == 'cosine':
        X_ = preprocessing.normalize(copy.deepcopy(X), axis=1)
    
    metric = defaultdict(list)
    for n_clusters in range(n_min, n_max+1):
        result = perform1DClustering(
            X, n_clusters, method=method, dist=dist, n_init=n_init,
            size_min=size_min, size_max=size_max)
        
        metric['cluster_number'].append(n_clusters)
        metric['sum_square_error'].append(np.array(result['cluster_inertia']))
        print(n_clusters, result['cluster_inertia'])
        metric['silhouette_score'].append(np.array(
            silhouette_score(X_,result['cluster_labels'],metric='euclidean')))
        metric['calinski_harabasz_score'].append(np.array(
            calinski_harabasz_score(X_, result['cluster_labels'])))
    return metric