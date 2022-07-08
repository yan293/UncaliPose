"""
   This scipt defines functions for clustering evaluation.
   Author: Yan Xu, CMU
   Update: Jul 01, 2022
"""

from sklearn.metrics.cluster import pair_confusion_matrix
import numpy as np


def getPurity(label_true, label_pred):
    '''
    Compute the purity score.
    '''
    clusters = np.unique(label_pred)
    label_true = np.reshape(label_true, (-1, 1))
    label_pred = np.reshape(label_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(label_pred == c)[0]
        labels_tmp = label_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    purity = np.sum(count) / label_true.shape[0]
    return purity


def getRandIndexAndFScore(label_true, label_pred, beta=1.):
    '''
    Compute the random index and the adjusted random index.
    '''
    (tn, fp), (fn, tp) = pair_confusion_matrix(label_true, label_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    if (tp+fn)*(fn+tn)+(tp+fp)*(fp+tn) == 0:
        ari = np.nan
    else:
        ari = 2.*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_score = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_score


def getClustMetrics(label_true, label_pred):
    '''
    Get all clustering metrics and return a metric dictionary.
    '''
    purity = getPurity(label_true, label_pred)
    ri, ari, f_score = getRandIndexAndFScore(label_true, label_pred, beta=1.)
    return { 'Purity': purity, 'RI': ri, 'ARI': ari, 'F_Score': f_score}
