import numpy as np


__all__ = [
    'mean_absolute_percentage_error',
]

def mean_absolute_percentage_error(y_true, y_pred):
    diff = np.abs(
        (y_true - y_pred) /
        np.clip(np.abs(y_true), np.finfo(float).eps, None)
    )
    return np.round(100. * np.mean(diff, axis=-1), 2)
