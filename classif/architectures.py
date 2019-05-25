
# keras
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D,\
    concatenate, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation,\
    Bidirectional, TimeDistributed, GRU, Reshape, Permute, GlobalMaxPooling2D

# from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

import keras.backend as K
from keras.models import Model

from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
# from keras.optimizers import SGD
# from attention import Attention

# =====================================================================================
"""
for now, hardcoding params in the functions
then, params:
-arc_id
-params_learn
-params_arch
"""


def get_model_wf_basic():
    """end-to-end: input is waveform. single shaped filters
        inputting 2s of sound
        """
    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))
    img_1 = Conv1D(16, kernel_size=9, activation=activations.relu, padding="valid")(inp)
    img_1 = Conv1D(16, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=16)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Conv1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu)(img_1)
    dense_1 = Dense(1028, activation=activations.relu)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

    # Conv2D: shape (batch, channel,time, freq) = (x, 1, time, 1)
    # input_shape = (1, int(params_extract.get('snip_len')), 1)
    # channel_axis = 1


def get_1d_dummy_model(params_learn=None, params_extract=None):

    n_class = params_learn.get('n_classes')
    # data_format='channels_first' does not work
    # Conv1D: "channels_first" corresponds to inputs with shape (batch, channels, length).
    # input_shape = (1, 100)
    # inp = Input(shape=input_shape)
    # x = Conv1D(4, kernel_size=9, activation=activations.relu, data_format='channels_first', padding='same')(inp)

    # hence we use channels_last
    # "channels_last" corresponds to inputs with shape (batch, length, channels) (default format for temporal data)
    input_shape = (params_extract.get('snip_len'), 1)
    inp = Input(shape=input_shape)
    x = Conv1D(4, kernel_size=9, activation=activations.relu, padding='same')(inp)

    x = MaxPool1D(pool_size=2048)(x)
    x = Flatten()(x)
    out = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    return model


def get_model_wf_lee(params_learn=None, params_extract=None):

    n_class = params_learn.get('n_classes')
    # we use channels_last
    # "channels_last" corresponds to inputs with shape (batch, length, channels) (default format for temporal data)
    input_shape = (params_extract.get('snip_len'), 1)
    inp = Input(shape=input_shape)
    channel_axis = 2

    x = Conv1D(128, kernel_size=3, strides=3, padding="same")(inp)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    # --
    x = Conv1D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    x = Conv1D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    # -- 6 layers of 256filters of len3 (we do 5)
    x = Conv1D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    x = Conv1D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    x = Conv1D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    x = Conv1D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    x = Conv1D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    # -- one of 512
    x = Conv1D(512, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)

    # -- one of 512 with no pooling
    x = Conv1D(512, kernel_size=1, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    # --drop
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    #

    out = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    return model


# ============================================================================T_F based models
# ============================================================================T_F based models
# ============================================================================T_F based models
#
#

def build_model_tf_basic(params_learn=None, params_extract=None):

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    channel_axis = 1
    n_class = params_learn.get('n_classes')
    melgram_input = Input(shape=input_shape)

    m_size = 3
    n_size = 70
    n_filters = 4
    maxpool_const = 4

    x = Conv2D(n_filters, (m_size, n_size),
                      padding='same',  # fmap has same size as input
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-5),
                      data_format='channels_first')(melgram_input)

    x = BatchNormalization(axis=channel_axis)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(int(params_extract.get('patch_len')//maxpool_const), params_extract.get('n_mels')), data_format="channels_first")(x)
    x = Flatten()(x)

    # x = Dropout(0.5)(x)
    # x = Dense(41, activation='softmax')(x)

    x = Dense(n_class,
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-5),
        activation='softmax',
        name='prediction')(x)

    model = Model(melgram_input, x)

    return model


def get_model_tf_js(params_learn=None, params_extract=None):

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    channel_axis = 1
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # pre-activation
    if params_learn.get('preactivation') == 1 or params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 2 or params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)

    # l1
    spec_x = Conv2D(24, (5, 5),
                      padding='same',  # fmap has same size as input
                      kernel_initializer='he_normal',
                      data_format='channels_first')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)
    spec_x = Activation('relu')(spec_x)

    if params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)

    # l2
    spec_x = Conv2D(48, (5, 5),
                      padding='same',  # fmap has same size as input
                      kernel_initializer='he_normal',
                      data_format='channels_first')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)
    spec_x = Activation('relu')(spec_x)

    if params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)

    # l3 (no pooling)
    spec_x = Conv2D(48, (5, 5),
                      padding='same',  # fmap has same size as input
                      kernel_initializer='he_normal',
                      data_format='channels_first')(spec_x)
    # x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(x)
    spec_x = Activation('relu')(spec_x)

    # before FC
    spec_x = Flatten()(spec_x)

    # dropout and dense_1
    spec_x = Dropout(0.5)(spec_x)
    spec_x = Dense(64,
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-3),
        activation='relu',
        name='dense_1')(spec_x)

    # dropout and dense_softmax
    spec_x = Dropout(0.5)(spec_x)
    out = Dense(n_class,
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-3),
        activation='softmax',
        name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out)

    return model


def get_model_tf_js_tidy(params_learn=None, params_extract=None):
    """
    This is JS version but following the more common order of:
    -Conv
    -BN
    -non-linearity
    -MP

    also allows preactivation

    :param params_learn:
    :param params_extract:
    :return:
    """
    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    channel_axis = 1
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # pre-activation
    if params_learn.get('preactivation') == 1 or params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 2 or params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)

    # l1
    spec_x = Conv2D(24, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(params_learn.get('dropout_prob'))(spec_x)

    if params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)

    # l2
    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(params_learn.get('dropout_prob'))(spec_x)

    if params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)

    # l3 (no pooling)
    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    # x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(x)

    # before FC
    spec_x = Flatten()(spec_x)

    # dropout and dense_1
    spec_x = Dropout(0.5)(spec_x)
    spec_x = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(spec_x)

    # dropout and dense_softmax
    spec_x = Dropout(0.5)(spec_x)
    out = Dense(n_class,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out)

    return model


def get_model_tf_kong_cnnbase18(params_learn=None, params_extract=None):
    """
    This is a keras version of the Kong DCASE 2018 cross-task baseline:
    https://github.com/qiuqiangkong/dcase2018_task2/blob/master/pytorch/models_pytorch.py

    described in:
    http://dcase.community/documents/challenge2018/technical_reports/DCASE2018_Baseline_87.pdf

    -Conv
    -BN
    -non-linearity
    -MP

    :param params_learn:
    :param params_extract:
    :return:
    """
    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    channel_axis = 1
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # pre-activation
    if params_learn.get('preactivation') == 1 or params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 2 or params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)


    # vip block1 - l1
    spec_x = Conv2D(64, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    # block1 - l2
    spec_x = Conv2D(64, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.3)(spec_x)

    if params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)


    # vip block2 - l3
    spec_x = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    # block2 - l4
    spec_x = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.3)(spec_x)

    # vip block3 - l5
    spec_x = Conv2D(256, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    # block3 - l6
    spec_x = Conv2D(256, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)

    # vip block4 - l7
    spec_x = Conv2D(512, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    # block4 - l8
    spec_x = Conv2D(512, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)

    # ================================================= classification block
    spec_x = GlobalMaxPooling2D(data_format="channels_first")(spec_x)

    # no need to flatten as we have a vector due to global
    # spec_x = Flatten()(spec_x)

    # dropout and dense_softmax
    out = Dense(n_class,
                kernel_initializer='he_normal',
                activation='softmax',
                name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out)

    return model

