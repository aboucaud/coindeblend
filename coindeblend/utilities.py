import os
import numpy as np

from keras import backend as K

__all__ = [
    'flux2mag',
    'plot_samples_one_hot',
    'save_model',
    'jaccard_coef'
]

# TF order (n_entries, shape_0, shape_1, n_channels)
AXIS = [1, 2, 3]  # all but n_entries
# Epsilon
EPS = 1e-12


def flux2mag(flux, zeropoint):
    return -2.5 * np.log10(flux) + zeropoint


def jaccard_coef(y_true, y_pred):
    """
    Compute the jaccard coefficient between two sets of images
    """
    intersection = K.sum(y_true * y_pred, axis=AXIS)
    sum_ = K.sum(y_true + y_pred, axis=AXIS)

    jac = (intersection + EPS) / (sum_ - intersection + EPS)

    return K.mean(jac)


def plot_samples_one_hot(labels, output_file=False):
    """Plot one hot encoded labels

    labels : array_like
        array of shape (num_samples, x, y, num_onehots)

    """
    import matplotlib.pyplot as plt

    if labels.ndim != 4:
        print("Incorrect input size - should be",
              "(num_samples, x, y, num_onehots)")
    num_samples, _, _, num_onehots = labels.shape
    fig_size = (4 * num_onehots, 4 * num_samples)
    fig, ax = plt.subplots(nrows=num_samples, ncols=num_onehots,
                           sharex=True, sharey=True, figsize=fig_size)
    for i in range(num_samples):
        for j in range(num_onehots):
            ax[i, j].imshow(labels[i, ..., j], aspect="auto")
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file)
    else:
        plt.show()


def plot_one_hot_eval(y_true, y_pred, output_file=False):
    """Plot one hot encoded labels

    labels : array_like
        array of shape (num_samples, x, y, num_onehots)

    """
    import matplotlib.pyplot as plt

    assert y_true.ndim == 3
    assert y_pred.ndim == 3
    fig_size = (16, 12)
    fig, ax = plt.subplots(nrows=3, ncols=4,
                           sharex=True, sharey=True, figsize=fig_size)
    for j in range(4):
        ax[0, j].imshow(y_true[..., j], aspect="auto")
        ax[1, j].imshow(y_pred[..., j], aspect="auto")
        ax[2, j].imshow(np.abs(y_true - y_pred)[..., j],
                        aspect="auto", cmap='gray')
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file)
        plt.close(fig)
    else:
        plt.show()


def save_model(model, cross, directory='.'):
    modeldir = os.path.join(directory, 'models')
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    json_string = model.to_json()
    json_name = 'architecture_' + cross + '.json'
    with open(os.path.join(modeldir, json_name), 'w') as arch:
        arch.write(json_string)

    weight_name = 'model_weights_' + cross + '.h5'
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)
