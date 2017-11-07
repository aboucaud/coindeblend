from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate

__all__ = ['UNet']


def UNet(input_shape, init_filt_size=64):
    """
    U-Net image segmentation model

    Parameters
    ----------
    input_shape : tuple of ints
        Shape of the input images
        The order depends on which keras backend is used.
            - TensorFlow => (row_size, col_size, channels)
            - Theano => (channels, row_size, col_size)
    init_filt_size : int, optional
        Size of the first filter (default is 64)
        It determines automatically the size of the next filters

    Returns
    -------
    model : Keras model

    References
    ----------
    https://arxiv.org/abs/1505.04597v1

    """
    img_input = Input(shape=input_shape)

    nfilt1 = init_filt_size
    nfilt2 = nfilt1 * 2
    nfilt3 = nfilt2 * 2

    # Block 1
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x_1a = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
                  name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),
                     name='block1_pool')(x_1a)

    # Block 2
    x = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(x)
    x_2a = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
                  name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),
                     name='block2_pool')(x_2a)

    # Block 3
    x = Conv2D(nfilt3, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(x)
    x = Conv2D(nfilt3, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x_2b = Conv2DTranspose(nfilt2, (2, 2), strides=(2, 2),
                           input_shape=(None, 23, 23, 1),
                           name='block3_deconv1')(x)

    # Deconv Block 1
    x = concatenate([x_2a, x_2b])
    x = Conv2D(nfilt2, kernel_size=(3, 3), activation='relu',
               padding='same', name='dblock1_conv1')(x)
    x = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
               name='dblock1_conv2')(x)
    x_1b = Conv2DTranspose(nfilt1, kernel_size=(2, 2), strides=(2, 2),
                           name='dblock1_deconv')(x)

    # Deconv Block 2
    x = concatenate([x_1a, x_1b], input_shape=(None, 92, 92, None),
                    name='dbock2_concat')
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='dblock2_conv1')(x)
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='dblock2_conv2')(x)
    # Output convolution.
    x = Conv2D(4, (1, 1), activation=None, padding='same',
               name='dblock2_conv3')(x)

    # Create model
    model = Model(img_input, x, name='UNet')

    return model
