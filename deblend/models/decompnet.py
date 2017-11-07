from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import SimpleRNN
from keras.layers import Reshape
from keras.layers import Flatten
from keras.backend import is_keras_tensor
from keras.engine.topology import get_source_inputs

__all__ = ['category_decomposition_net', 'instance_decomposition_net']


def category_decomposition_net(input_tensor=None, input_shape=None):
    """
    """
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # conv(1,32,5,2)
    x = Conv2D(32, (5, 5), strides=2)(img_input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # conv(32,32,5,2)
    x = Conv2D(32, (5, 5), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # conv(32,64,3,2)
    x = Conv2D(64, (3, 3), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # conv(64,64,3,1)
    x = Conv2D(64, (3, 3), strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # conv(64,64,3,2)
    x = Conv2D(64, (3, 3), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # conv(64,64,3,1)
    x = Conv2D(64, (3, 3), strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # deconv(64,64,3,2)
    x = Conv2DTranspose(64, (3, 3), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # deconv(64,64,3,2)
    x = Conv2DTranspose(64, (3, 3), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # deconv(64,64,3,2)
    x = Conv2DTranspose(64, (3, 3), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # deconv(64,1,1,1)
    x = Conv2DTranspose(1, (1, 1), strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name="DecompNet_part1")

    return model


def instance_decomposition_net(input_tensor=None, input_shape=None):
    """
    """
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # conv(1,32,5,2)-
    x = Conv2D(32, (5, 5), strides=2)(img_input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # conv(32,32,3,2)
    x = Conv2D(32, (3, 3), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # conv(32,32,3,2)
    x = Conv2D(32, (3, 3), strides=2)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # x = Reshape()(x)
    # rnn-fc
    x = SimpleRNN(2048)(x)
    x = Activation('relu')(x)
    # rnn-fc
    x = SimpleRNN(2048)(x)
    x = Activation('relu')(x)
    # deconv(32,32,3,1)
    x = Conv2DTranspose(32, (3, 3), strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # deconv(32,32,3,1)
    x = Conv2DTranspose(64, (3, 3), strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # deconv(32,32,5,1)
    x = Conv2DTranspose(64, (5, 5), strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # deconv(64,3,1,1)
    x = Conv2DTranspose(1, (1, 1), strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name="DecompNet_part2")

    return model
