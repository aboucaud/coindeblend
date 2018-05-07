from __future__ import division

import numpy as np
from keras import backend as K

from .utilities import jaccard_coef

__all__ = ['IOU', 'global_accuracy', 'mean_IOU', 'jaccard_coef_int', 'iou']


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


def iou_bitmap(y_true, y_pred, verbose=False):
    """
    Compute the IoU between two arrays

    If the arrays are probabilities (floats) instead of predictions (integers
    or booleans) they are automatically rounded to the nearest integer and
    converted to bool before the IoU is computed.

    Parameters
    ----------
    y_true : ndarray
        array of true labels
    y_pred : ndarray
        array of predicted labels
    verbose : bool (optional)
        print the intersection and union separately

    Returns
    -------
    float :
        the intersection over union (IoU) value scaled between 0.0 and 1.0

    """
    EPS = np.finfo(float).eps

    # Make sure each pixel was predicted e.g. turn probability into prediction
    if y_true.dtype in [np.float32, np.float64]:
        y_true = y_true.round().astype(bool)

    if y_pred.dtype in [np.float32, np.float64]:
        y_pred = y_pred.round().astype(bool)

    # Reshape to 1d
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Compute intersection and union
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true + y_pred)
    jac = (intersection + EPS) / (sum_ - intersection + EPS)

    if verbose:
        print('Intersection:', intersection)
        print('Union:', sum_ - intersection)

    return jac


def iou(y_true, y_pred):
    iou_list = [iou_bitmap(yt, yp)
                for (yt, yp) in zip(y_true, y_pred)]
    return np.mean(iou_list)
