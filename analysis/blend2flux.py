from __future__ import division

import os
from math import ceil

import numpy as np

import logging

from sklearn.utils import Bunch
from keras.models import Sequential, Model
from keras.layers import (Conv2D, Dropout, Input, concatenate, MaxPooling2D,
                          Conv2DTranspose, UpSampling2D)
from keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
                             TensorBoard, LambdaCallback, CSVLogger)
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise

from deblend.models import UNet_modular
from deblend.models import SeqStack_modular_clean
from deblend.scores import jaccard_coef_int, iou


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
                 filename=None, maindir=None, seed=42, plot_history=False, plot_results=False):
        self.model_, self.params_model_ = self._build_model()
        self.batch_size = batch_size
        self.epoch = epoch
        self.model_check_point = model_check_point
        self.filename = filename
        self.maindir = maindir
        self.seed = seed
        self._plot_history = plot_history
        self._plot_results = plot_results
        self.log = logging.getLogger(__name__)

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
        if X.ndim == 3:
            X = np.expand_dims(X, -1)

            # create the callbacks to get during fitting
        callbacks = self._build_callbacks()

        # fit the model
        history = self.model_.fit(X, y,
                                  epochs=self.epoch,
                                  batch_size=self.batch_size,
                                  callbacks=callbacks,
                                  validation_split=0.3)

        if self._plot_history:
            self.plot_history(history)


    def predict(self, X):
        if X.ndim == 3:
            X = np.expand_dims(X, -1)
        return self.model_.predict(X)

    def predict_score(self, X, y_true):
        if X.ndim == 3:
            X = np.expand_dims(X, -1)

        return self.model_.evaluate(X, y_true)[0]

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

    def plot_results(self, X, Y):
        if self._plot_results:
            import matplotlib
            matplotlib.use('Agg')
            import pylab as plt

            self.log.info(" => Plotting results..")

            Y_pred = self.model_.predict(np.expand_dims(X, axis=-1), batch_size=self.batch_size)

            gal = 0
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.loglog(Y[:, gal], Y_pred[:, gal], ',')
            ax.plot([0, 1600], [0, 1600], 'k--', alpha=0.3, lw=1)
            ax.set_title(f'{self.filename} - central galaxy')
            ax.set_xlim(10, 2000)
            ax.set_ylim(10, 2000)
            ax.set_xlabel('True flux')
            ax.set_ylabel('Predicted flux')
            fig.tight_layout()
            fig.savefig(f'jobs/{self.filename}/plots/{self.filename}-central_gal_flux.png')
            plt.close(fig)

            gal = 1
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.loglog(Y[:, gal], Y_pred[:, gal], ',')
            ax.plot([0, 1600], [0, 1600], 'k--', alpha=0.3, lw=1)
            ax.set_title(f'{self.filename} - companion galaxy')
            ax.set_xlim(10, 2000)
            ax.set_ylim(10, 2000)
            ax.set_xlabel('True flux')
            ax.set_ylabel('Predicted flux')
            fig.tight_layout()
            fig.savefig(f'jobs/{self.filename}/plots/{self.filename}-companion_gal_flux.png')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.hist(Y - Y_pred, bins=200);
            ax.set_xlim(-200, 200);
            ax.set_title(f'{self.filename}-histogram_of_residuals')
            fig.tight_layout()
            fig.savefig(f'jobs/{self.filename}/plots/{self.filename}-histogram_of_residuals.png')
            plt.close(fig)

        else:
            self.log.info(" => Not plotting, change to plot_results=True for plotting")


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
        params_model.output_channels = 2
        params_model.depth = 2  #3
        params_model.init_filt_size = 512
        params_model.init_filt_dense = 512
        params_model.depth_dense = 2
        params_model.dropout_rate = 0.2

        # LOSS
        # this is basically the metric for optimizing your model
        # this needs to be selected in accordance to the task you want to achieve
        # params_model.keras_loss = 'binary_crossentropy'
        params_model.keras_loss = 'mean_absolute_percentage_error'  #'categorical_crossentropy' #'mse'

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
        params_model.early_stopping = True
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

        model = SeqStack_modular_clean(
            input_shape=(params_model.img_rows,
                         params_model.img_cols,
                         params_model.img_channels),
            filt_size=params_model.init_filt_size,
            depth=params_model.depth,
            filt_dense=params_model.init_filt_dense,
            depth_dense=params_model.depth_dense)

        optimizer = Adam(lr=params_model.lr)


        model.compile(optimizer=optimizer,
                      loss=params_model.keras_loss,
                      metrics=['acc'])

        return model, params_model

    def _build_callbacks(self):
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
            logdir = os.path.join(self.logdir, f"{self.filename}")
            callbacks.append(
                TensorBoard(log_dir=logdir,
                            write_grads=self.params_model_.tb_write_grads,
                            batch_size=self.batch_size)
            )

        return callbacks


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
    datadir = os.path.join(maindir, "data_good")
    workdir = os.path.join(maindir, "jobs", job_id)
    logfile = os.path.join(workdir, "run.log")
    resfile = os.path.join(maindir, 'results.csv')

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
    Y_train = np.load(os.path.join(datadir, 'train_flux.npy'), mmap_mode='r')
    X_test = np.load(os.path.join(datadir, 'test_images.npy'), mmap_mode='r')
    Y_test = np.load(os.path.join(datadir, 'test_flux.npy'), mmap_mode='r')

    log.info(" => Creating object detector")
    obj = ObjectDetector(epoch=200,
                         model_check_point=True,
                         filename=job_id,
                         maindir=workdir,
                         plot_history=False,
                         plot_results=True)


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

    # Results file
    header = ','.join(["job_id", "train_time", "test_time", "score"])
    scoreline = f"{job_id},{traintime:.2f},{testtime:.2f},{score}"

    if not os.path.exists(resfile):
        with open(resfile, 'a') as f:
            print(header, file=f)

    with open(resfile, 'a') as f:
        print(scoreline, file=f)

    # Plotting for plot_results=True
    obj.plot_results(X_test, Y_test)

if __name__ == '__main__':
    main()
