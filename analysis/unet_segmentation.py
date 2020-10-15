from __future__ import division

import io
import os
import logging
from math import ceil

import numpy as np

from sklearn.utils import Bunch
from keras.models import Sequential, Model
from keras.layers import (Conv2D, Dropout, Input, concatenate, MaxPooling2D,
                          Conv2DTranspose, UpSampling2D)
from keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
                             TensorBoard, LambdaCallback, CSVLogger)
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise

from coindeblend.models import UNet_modular
from coindeblend.scores import jaccard_coef_int, iou

import matplotlib.pyplot as plt
import tensorflow as tf


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    buf.close()
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


class ObjectDetector(object):
    """Object detector.

    Parameters
    ----------
    batch_size : int, optional
        The batch size used during training. Set by default to 32 samples.

    epoch : int, optional
        The number of epoch for which the model will be trained. Set by default
        to 50 epochs.

    model_check_point : bool, optional
        Whether to create a callback for intermediate models.

    Attributes
    ----------
    model_ : object
        The DNN model.

    params_model_ : Bunch dictionary
        All hyper-parameters to build the DNN model.

    """

    def __init__(self, batch_size=32, epoch=10, model_check_point=True,
                 filename=None, maindir=None, seed=42, plot_history=False,
                 display_img=None):
        self.model_, self.params_model_ = self._build_model()
        self.batch_size = batch_size
        self.epoch = epoch
        self.model_check_point = model_check_point
        self.filename = filename
        self.maindir = maindir
        self.seed = seed
        self._plot_history = plot_history
        self.log = logging.getLogger(__name__)
        self.test_display_img = display_img

        self._init_repos()
        self._write_info()

    def _write_info(self):
        self.log.info("")
        self.log.info("\tModel summary")
        self.log.info("\t=============")
        self.model_.summary(print_fn=lambda x: self.log.info(f"\t{x}"))
        self.log.info("")
        self.log.info("\tParameters")
        self.log.info("\t==========")
        for k, v in self.params_model_.items():
            self.log.info(f"\t{k}: {v}")
        self.log.info("")

    def _init_repos(self):
        self.weightdir = os.path.join(self.maindir, 'weights')
        self.plotdir = os.path.join(self.maindir, 'plots')
        self.logdir = os.path.join(os.getenv('COIN'), 'projects', 'logs')

        os.makedirs(self.weightdir, exist_ok=True)
        os.makedirs(self.plotdir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)

    def load_weights(self, weights_file):
        self.model_.load_weights(weights_file)

    def fit(self, X, y):
        # build the box encoder to later encode y to make usable in the model
        train_dataset = BatchGeneratorBuilder(X, y)
        train_generator, val_generator, n_train_samples, n_val_samples = \
            train_dataset.get_train_valid_generators(
                batch_size=self.batch_size,
                valid_ratio=self.params_model_.valid_ratio)

        # create the callbacks to get during fitting
        callbacks = self._build_callbacks()

        # fit the model
        history = self.model_.fit_generator(
            generator=train_generator,
            steps_per_epoch=ceil(n_train_samples / self.batch_size),
            epochs=self.epoch,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=ceil(n_val_samples / self.batch_size))

        if self._plot_history:
            self.plot_history(history)

    def predict(self, X):
        if X.ndim == 3:
            X = np.expand_dims(X, -1)
        return self.model_.predict(X)

    def predict_score(self, X, y_true):
        if X.ndim == 3:
            X = np.expand_dims(X, -1)
        if y_true.ndim == 3:
            y_true = np.expand_dims(y_true, -1)

        y_pred = self.model_.predict(X)
        return iou(y_true, y_pred)

    def plot_history(self, history):
        import matplotlib.pyplot as plt

        self.log.info(" => Creating plots..")

        plt.figure()
        plt.semilogy(history.epoch, history.history['loss'], label='train loss')
        plt.semilogy(
            history.epoch, history.history['val_loss'], label='valid loss')
        plt.title('Loss history')
        plt.legend()
        plt.tight_layout()
        loss_plot_file = os.path.join(
            self.plotdir, f"{self.filename}_train_history.png")
        plt.savefig(loss_plot_file)
        plt.close()

        self.log.info(f"\tplot saved in {loss_plot_file}")

        plt.figure()
        plt.plot(history.epoch, history.history['acc'], label='train accuracy')
        plt.plot(history.epoch, history.history['val_acc'], label='valid accuracy')
        plt.title('Accuracy history')
        plt.legend()
        plt.tight_layout()
        accuracy_plot_file = os.path.join(
            self.plotdir, f"{self.filename}_train_accuracy.png")
        plt.savefig(accuracy_plot_file)
        plt.close()

        self.log.info(f"\tplot saved in {accuracy_plot_file}")


    ###########################################################################
    # Setup model

    @staticmethod
    def _init_params_model():
        params_model = Bunch()

        # image and class parameters
        params_model.img_rows = 128
        params_model.img_cols = 128
        params_model.img_channels = 1

        # ARCHITECTURE PARAMS
        # they depend exclusively on your model
        # the names can be changed since they will only be called by your model
        params_model.output_channels = 3
        params_model.depth = 6
        params_model.init_filt_size = 64
        params_model.dropout_rate = 0.3

        # LOSS
        # this is basically the metric for optimizing your model
        # this needs to be selected in accordance to the task you want to achieve
        params_model.keras_loss = 'binary_crossentropy'
        # params_model.keras_loss = 'mse'

        # OPTIMIZER PARAMS
        # these values are good starting point
        # you should not change them during the first runs.
        params_model.lr = 1e-4
        params_model.beta_1 = 0.9
        params_model.beta_2 = 0.999
        params_model.epsilon = 1e-08
        params_model.decay = 5e-05

        params_model.valid_ratio = 0.2

        # callbacks parameters
        # params_model.early_stopping = True
        params_model.early_stopping = False
        params_model.es_patience = 12
        params_model.es_min_delta = 0.001

        params_model.reduce_learning_rate = True
        params_model.lr_patience = 5
        params_model.lr_factor = 0.5
        params_model.lr_min_delta = 0.001
        params_model.lr_cooldown = 2

        params_model.tensorboard = True
        params_model.tb_write_grads = True

        return params_model

    def _build_model(self):

        # load the parameter for the SSD model
        params_model = self._init_params_model()

        #######################################################################
        #
        # --- CHANGE HERE ---
        #

        # The deep neural network model can be imported from an external file
        # like here or be defined right below.

        model = UNet_modular(
            input_shape=(params_model.img_rows,
                         params_model.img_cols,
                         params_model.img_channels),
            output_channels=params_model.output_channels,
            depth=params_model.depth,
            filt_size=params_model.init_filt_size,
            dropout_rate=params_model.dropout_rate)

        optimizer = Adam(lr=params_model.lr)

        #
        #
        #######################################################################


        model.compile(optimizer=optimizer,
                      loss=params_model.keras_loss,
                      metrics=['acc'])

        return model, params_model

    def _build_callbacks(self):
        logdir = os.path.join(self.logdir, f"{self.filename}")

        callbacks = []

        epoch_logger = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: self.log.info(
                f"\t\tStarting Epoch {epoch}/{self.epoch}"))
        callbacks.append(epoch_logger)

        csv_logger = CSVLogger(os.path.join(self.plotdir, 'history.csv'))
        callbacks.append(csv_logger)

        if self.model_check_point:
            wdir = os.path.join(self.weightdir, f'{self.filename}_weights_best.h5')
            callbacks.append(
                ModelCheckpoint(wdir,
                                monitor='val_loss',
                                save_best_only=True,
                                save_weights_only=True,
                                period=1,
                                verbose=1))
        # add early stopping
        if self.params_model_.early_stopping:
            callbacks.append(
                EarlyStopping(monitor='val_loss',
                              min_delta=self.params_model_.es_min_delta,
                              patience=self.params_model_.es_patience,
                              verbose=1))

        # reduce learning-rate when reaching plateau
        if self.params_model_.reduce_learning_rate:
            callbacks.append(
                ReduceLROnPlateau(monitor='val_loss',
                                  factor=self.params_model_.lr_factor,
                                  patience=self.params_model_.lr_patience,
                                  cooldown=self.params_model_.lr_cooldown,
                                  # min_delta=self.params_model_.lr_min_delta,
                                  verbose=1))
        if self.params_model_.tensorboard:

            callbacks.append(
                TensorBoard(log_dir=logdir,
                            write_grads=self.params_model_.tb_write_grads,
                            batch_size=self.batch_size)
            )

        return callbacks


###############################################################################
# Batch generator

class BatchGeneratorBuilder(object):
    """A batch generator builder for generating batches of images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).

    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image data to train on
    y_array : vector of int
        vector of object labels corresponding to `X_array`

    """
    def __init__(self, X_array, y_array):
        self.X_array = X_array
        self.y_array = y_array
        self.nb_examples = len(X_array)
        self.X_single_channel = X_array.ndim == 3
        self.y_single_channel = y_array.ndim == 3


    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=32):
        if indices is None:
            indices = np.arange(self.nb_examples)
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:
            X = self.X_array[indices]
            y = self.y_array[indices]

            # converting to float needed?
            X = np.array(X, dtype='float32')
            y = np.array(y, dtype='float32')

            # Yielding mini-batches
            for i in range(0, len(X), batch_size):

                if self.X_single_channel:
                    X_batch = [np.expand_dims(img, -1)
                               for img in X[i:i + batch_size]]
                else:
                    X_batch = [img for img in X[i:i + batch_size]]

                if self.y_single_channel:
                    y_batch = [np.expand_dims(seg, -1)
                               for seg in y[i:i + batch_size]]
                else:
                    y_batch = [seg for seg in y[i:i + batch_size]]

                for j in range(len(X_batch)):

                    # flip images
                    if np.random.randint(2):
                        X_batch[j] = np.flip(X_batch[j], axis=0)
                        y_batch[j] = np.flip(y_batch[j], axis=0)

                    if np.random.randint(2):
                        X_batch[j] = np.flip(X_batch[j], axis=1)
                        y_batch[j] = np.flip(y_batch[j], axis=1)

                    # TODO add different data augmentation steps

                yield np.array(X_batch), np.array(y_batch)


def main():
    import sys
    import time

    try:
        extra_arg = sys.argv[1]
    except IndexError:
        extra_arg = ""

    if extra_arg in ['--help', '-h']:
        print("\nUsage:\n"
              f"\tpython {sys.argv[0]} <name/id>")
        sys.exit()

    filename = os.path.splitext(sys.argv[0])[0]
    job_id = "_".join([f"{filename}"] + sys.argv[1:])

    maindir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.getenv("COINBLEND_DATADIR")
    workdir = os.path.join(maindir, "jobs", job_id)
    logfile = os.path.join(workdir, "run.log")
    resfile = os.path.join(maindir, 'results.csv')
    modelfile = os.path.join(workdir, "model.json")
    fullmodelfile = os.path.join(workdir, "fullmodel.h5")
    predictionfile = os.path.join(workdir, "test_predictions.npy")

    if os.path.exists(workdir):
        print("\n --== WARNING ==--")
        print(f"Directory 'jobs/{job_id}' already existing")
        print("Job aborting..")
        sys.exit()

    os.makedirs(workdir)

    logging.basicConfig(filename=logfile, level=logging.INFO)
    log = logging.getLogger(__name__)

    log.info(" => Reading data")

    X_train = np.load(os.path.join(datadir, 'train_images.npy'), mmap_mode='r')
    Y_train = np.load(os.path.join(datadir, 'train_masks.npy'), mmap_mode='r')
    X_test = np.load(os.path.join(datadir, 'test_images.npy'), mmap_mode='r')
    Y_test = np.load(os.path.join(datadir, 'test_masks.npy'), mmap_mode='r')

    test_idx = np.random.randint(0, len(X_test))
    display_tuple = (X_test[test_idx], Y_test[test_idx])

    ###########################################################################
    #
    # --- CHANGE HERE ---
    #

    log.info(" => Creating object detector")
    obj = ObjectDetector(batch_size=32,
                         epoch=200,
                         model_check_point=True,
                         filename=job_id,
                         maindir=workdir,
                         plot_history=False,
                         display_img=display_tuple)

    #
    ###########################################################################

    t0 = time.time()
    log.info(" => Training...")
    obj.fit(X_train, Y_train)

    t1 = time.time()
    log.info(" => Testing...")
    score = obj.predict_score(X_test, Y_test)

    t2 = time.time()

    traintime = t1 - t0
    testtime = t2 - t1

    log.info("")
    log.info("\tResults")
    log.info("\t=======")
    log.info(f"\ttrain_time: {traintime:.1f} sec - {traintime / 60:.1f} min")
    log.info(f"\ttest_time: {testtime:.1f} sec - {testtime / 60:.1f} min")
    log.info(f"\tscore: {score*100:.2f}/100")


    obj.model_.save(fullmodelfile)
    np.save(
        predictionfile,
        np.squeeze(
            obj.model_.predict(
                np.expand_dims(X_test, -1)
            )
        )
    )


if __name__ == '__main__':
    main()
