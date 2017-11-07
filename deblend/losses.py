from keras import backend as K

from .utilities import jaccard_coef

__all__ = ['jaccard_coef_loss']


def jaccard_coef_loss(y_true, y_pred):
    """
    Loss based on the jaccard coefficient, regularised with
    binary crossentropy

    Notes
    -----
    Found in https://github.com/ternaus/kaggle_dstl_submission

    """
    return (-K.log(jaccard_coef(y_true, y_pred)) +
            K.binary_crossentropy(y_pred, y_true))
