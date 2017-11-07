from __future__ import division

import numpy as np
from keras import backend as K

from .utilities import jaccard_coef

__all__ = ['IOU', 'global_accuracy', 'mean_IOU', 'jaccard_coef_int']


def jaccard_coef_int(y_true, y_pred):
    """
    """
    # Clip the y_pred between 0 and 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    return jaccard_coef(y_true, y_pred_pos)


def IOU(y_pred, y_true):
    """Compute the IOU for each class seperately"""
    # y_pred = y_pred.astype('bool')
    # y_true = y_true.astype('bool')
    intersection = np.sum(y_pred & y_true, axis=(0, 1, 2))
    union = np.sum(y_pred | y_true, axis=(0, 1, 2))

    if np.any(union == 0):
        wherezero = np.where(union == 0)
        union[wherezero] = intersection[wherezero]

    return intersection / union


def global_accuracy(y_pred, y_true):
    """Compute the percentage of pixels correctly labelled"""
    # y_pred = y_pred.astype('bool')
    # y_true = y_true.astype('bool')
    num_correct = np.sum(y_pred & y_true)
    num_total = np.product(y_true.shape[:-1])

    return num_correct / num_total


def mean_IOU(y_pred, y_true):
    """Compute the mean IOU taken over classes"""
    IoU = IOU(y_pred, y_true)
    return IoU.sum() / IoU.size
