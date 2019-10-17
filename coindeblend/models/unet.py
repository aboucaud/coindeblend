from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Dropout

__all__ = [
    'UNet',
    'UNet_modular',
    'unet_dropout', 
    ]


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


def unet_dropout(input_shape, output_channels, p):
    inputs = Input(input_shape)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(p)(conv1)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Dropout(p)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(p)(conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Dropout(p)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(p)(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(p)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(p)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Dropout(p)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(p)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(p)(conv5)

    up6 = concatenate([Conv2DTranspose(64, (3, 3), strides=(
        2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(p)(conv6)

    up7 = concatenate([Conv2DTranspose(32, (3, 3), strides=(
        2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(p)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(p)(conv7)

    up8 = concatenate([Conv2DTranspose(16, (3, 3), strides=(
        2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(p)(conv8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(p)(conv8)

    up9 = concatenate([Conv2DTranspose(8, (3, 3), strides=(
        2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(p)(conv9)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Dropout(p)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  # 3
    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def UNet_modular(input_shape, output_channels, filt_size, depth, 
                 activation='relu', dropout_rate=0.0, 
                 kernel_size=(3, 3), pool_size=(2, 2)):
    """
    """
    inputs = Input(shape=input_shape)

    x = inputs
    # Encoding
    convolutions = []
    for d in range(0, depth-1):
        fsize = filt_size * 2 ** d
        x = Conv2D(fsize, kernel_size, activation=activation, padding='same', name='block%d_conv1' % d)(x)
        x = Dropout(dropout_rate, name='block%d_dropout1' % d)(x)
        x = Conv2D(fsize, kernel_size, activation=activation, padding='same', name='block%d_conv2' % d)(x)
        convolutions.append((fsize, x))
        x = Dropout(dropout_rate, name='block%d_dropout2' % d)(x)
        x = MaxPooling2D(pool_size=pool_size, name='block%d_maxpool' % d)(x)

    # Middle layer
    x = Conv2D(filt_size * 2 ** (depth-1), kernel_size, activation=activation, padding='same', name='midblock_conv1')(x)
    x = Dropout(dropout_rate, name='midblock_dropout1')(x)
    x = Conv2D(filt_size * 2 ** (depth-1), kernel_size, activation=activation, padding='same', name='midblock_conv2')(x)
    x = Dropout(dropout_rate, name='midblock_dropout2')(x)

    # Decoding
    for d, (fsize, conv) in enumerate(reversed(convolutions)):
        x = Conv2DTranspose(fsize, (3, 3), strides=(2, 2), padding='same', name='deconvblock%d' % d)(x)
        x = concatenate([x, conv], axis=3)
        x = Conv2D(fsize, kernel_size, activation=activation, padding='same', name='dblock%d_conv1' % d)(x)
        x = Dropout(dropout_rate, name='dblock%d_dropout1' % d)(x)
        x = Conv2D(fsize, kernel_size, activation=activation, padding='same', name='dblock%d_conv2' % d)(x)
        x = Dropout(dropout_rate, name='dblock%d_dropout2' % d)(x)

    x = Conv2D(output_channels, (1, 1), activation='sigmoid', name='last_block')(x)
    
    model = Model(inputs=[inputs], outputs=[x])

    return model
