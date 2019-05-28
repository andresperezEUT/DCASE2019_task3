
# keras
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D,\
    Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation,\
    Bidirectional, TimeDistributed, GRU, Reshape, Permute, GlobalMaxPooling2D, GlobalAveragePooling2D, SeparableConv2D

from keras.layers import Average, Concatenate, Multiply

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
    n_filters = 1
    maxpool_const = 4

    x = Conv2D(n_filters, (m_size, n_size),
                      padding='same',  # fmap has same size as input
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-5),
                      data_format='channels_first')(melgram_input)

    # x = BatchNormalization(axis=channel_axis)(x)
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


def get_model_crnn_sa(params_crnn=None, params_learn=None, params_extract=None):
    """
    original:
    def get_model(data_in, data_out, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):

    adapted from
    https://github.com/sharathadavanne/multichannel-sed-crnn/blob/master/sed.py

    :param data_in:
    :param data_out:
    :param _cnn_nb_filt: # CNN filter size
    :param _cnn_pool_size: # Maxpooling across frequency *only*. Length of cnn_pool_size =  number of CNN layers
    :param _rnn_nb: # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
    :param _fc_nb: # Number of FC nodes.  Length of fc_nb =  number of FC layers
    :param dropout_rate: # Dropout after each layer
    :return:

    other:
    -using channels_first (given by the BN and by K.set_image_data_format('channels_first'))

    T_F input:
        -fixed length, to XXX frames of 40ms and 50% overlap
        -they used 256 frames, which means roughly 5 seconds?
        -originally: seq_len = 256 # Frame sequence length. Input to the CRNN.
        -for dcaseT4 and BAD, 10s audios, 500 frames
        -we use:
            -128 bands
            -patches de 1s = 50 frames, 2s = 100, 3s = 150, en modo varup

    CONVS:
        -using 3 layers. We could change that.
        -they all have the same number of filters (128). It could vary, but I guess they tried this.
        -filter shape is 3x3. how about other shapes? vertical? horizontal?
        -MaxPooling2D, BUT only in the freq domain, thus preserving the time (time, freq) = (1, target)
        -target could be 1 or 2, something small
        -Dropout in every conv layer

    RECS:
        -using 2 layers
        -GRU (not LSTM)
        -tanh
        -dropout

    DENSE:
        -TimeDistributed: This wrapper applies a layer to every temporal slice of an input.
        -The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension.
        -Consider a batch of 32 samples, where each sample is a sequence of 10 frames of 16 melbands.
            The batch input shape of the layer is then (32, 10, 16)
        -You can then use TimeDistributed to apply a Dense layer to each of the 10 timesteps, independently
        -The output will then have shape (32, 10, number of nodes in the Dense layer)

    """
    K.set_image_data_format('channels_first')

    # OLD
    # input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    # melgram_input = Input(shape=input_shape)
    n_class = params_learn.get('n_classes')

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))

    # make sure to understand this, channel_ok, time_ok, freq_ok. Does not include the batch axis.
    # spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # pre-activation
    if params_learn.get('preactivation') == 1 or params_learn.get('preactivation') == 3:
        spec_x = BatchNormalization(axis=1)(spec_x)
    elif params_learn.get('preactivation') == 2 or params_learn.get('preactivation') == 4:
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)

    # Conv layers
    for _i, _cnt in enumerate(params_crnn.get('cnn_pool_size')):
        spec_x = Conv2D(filters=params_crnn.get('cnn_nb_filt'), kernel_size=(3, 3), padding='same')(spec_x)
        # after a Conv2D layer with data_format="channels_first", set axis=1
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, params_crnn.get('cnn_pool_size')[_i]))(spec_x)
        spec_x = Dropout(params_crnn.get('dropout_rate'))(spec_x)

        if params_learn.get('preactivation') == 3:
            spec_x = BatchNormalization(axis=1)(spec_x)
        elif params_learn.get('preactivation') == 4:
            spec_x = BatchNormalization(axis=1)(spec_x)
            spec_x = Activation('relu')(spec_x)

    # seems here we have dimension:
    # channel (128), time (untouched, cause there was no pooling), freq (reduced by MP, by 20 times), hence 6 in freq

    # here we have 3 conv layers, now let's plug the rec layers:

    # Permutes the dimensions of the input according to a given pattern to (time, channel, freq)
    # Useful for e.g. connecting RNNs and convnets together.
    spec_x = Permute((2, 1, 3))(spec_x)

    # target_shape: Tuple of integers. Does not include the batch axis.
    # Output shape: (batch_size,) + target_shape
    # target_shape is (the original time dimension, whatever needed to match the dimensions)
    # the latter means concatenate the different activation maps to go:
    # -from a tensor of depth 128 activation maps of time,freq
    # -to a matrix of (time, 128*freq)
    # spec_x = Reshape((data_in.shape[-2], -1))(spec_x)
    spec_x = Reshape((params_extract.get('patch_len'), -1))(spec_x)

    for _r in params_crnn.get('rnn_nb'):
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=params_crnn.get('dropout_rate'), recurrent_dropout=params_crnn.get('dropout_rate'), return_sequences=True),
            merge_mode='mul')(spec_x)

    # dims here:
    # -time dim is preserved, untouched
    # -the other dim, 'freq', is 32 due to 32 nodes of the GRU, _r

    # plan A) add attention after RNN and direct to dense softmax
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Attention())
    # next add a Dense layer (for classification/regression) or whatever...

    if params_learn.get('attention'):
        spec_x = Attention()(spec_x)
        # todo which is the dimension here? time, _f
        # time steps must be maintained, but features? maybe too? if so, need to flatten.

        # -------------------------for simplicity we omit the FINAL FCs layers, and get directly to the softmax
        # apply a dense layer to every time frame
        # if u dont want to do SED, why apply TimeDistributed?, to add attention
        # for _f in _fc_nb:
        #     spec_x = TimeDistributed(Dense(_f))(spec_x)
        #     spec_x = Dropout(dropout_rate)(spec_x)
        # NOTE: I have not tried to put FC after attention, but I did try to Flatten after the Attention and it does now work
        # check the dimensions
        # todo which is the dimension here? time, _f

    else:
        for _f in params_crnn.get('fc_nb'):
            spec_x = TimeDistributed(Dense(_f))(spec_x)
            spec_x = Dropout(params_crnn.get('dropout_rate'))(spec_x)

        # apply a dense layer to every time frame, mapping it to the number of classes to predict
        # hence we predict class activity on per-frame basis
        # sigmoid for multilabel (will need thresholding in post-pro)

        # originally, we predicted multiple labels for every time slice, hence we can do SED.
        # this is useful for the case where we want to:
        # -determine class activity with timestamps with certain time resolution (hence TimeDistributed)
        # -several classes can overlap (ie multilabel, hence sigmoid & binary_crossentropy)
        # -several classes can occur in the same clip
        # now, the problem is different:
        # -no need for timestamps (hence paso de TimeDistributed)
        # OLD from kaggle18: - no concurrent classes, and only one can occur per file (ie multiclass, hence softmax & categorical_crossentropy)

        # we have 2D, need flatten
        # before FC
        spec_x = Flatten()(spec_x)

    # common to all: output layer
    spec_x = Dense(n_class)(spec_x)
    # out = Activation('sigmoid', name='weak_out')(spec_x)
    out = Activation('softmax',name='weak_out')(spec_x)
    # OLD
    # spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    # out = Activation('sigmoid', name='strong_out')(spec_x)
    # _model = Model(inputs=spec_start, outputs=out)
    # _model.compile(optimizer='Adam', loss='binary_crossentropy')
    # _model.summary()

    _model = Model(inputs=spec_start, outputs=out)
    # done in main
    # _model.compile(optimizer='Adam', loss='categorical_crossentropy')
    # _model.summary()
    return _model


def get_model_crnn_seld(params_crnn=None, params_learn=None, params_extract=None):
    # def get_model_crnn_seld(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
    #                                 rnn_size, fnn_size, weights):

    K.set_image_data_format('channels_first')

    n_class = params_learn.get('n_classes')

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))

    # make sure to understand this, channel_ok, time_ok, freq_ok. Does not include the batch axis.
    # spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_start = Input(shape=input_shape)
    spec_cnn = spec_start

    # CNN
    for i, convCnt in enumerate(params_crnn.get('cnn_pool_size')):
        spec_cnn = Conv2D(filters=params_crnn.get('cnn_nb_filt'), kernel_size=(3, 3), padding='same')(spec_cnn)
        # after a Conv2D layer with data_format="channels_first", set axis=1
        spec_x = BatchNormalization(axis=1)(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, params_crnn.get('cnn_pool_size')[i]))(spec_cnn)
        spec_cnn = Dropout(params_crnn.get('dropout_rate'))(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)  # model definition

    # seems here we have dimension:
    # channel (nb of filters), time (untouched, cause there was no pooling), freq (reduced by MP, by 20 times), hence 6 in freq

    # here we have 3 conv layers, now let's plug the rec layers:

    # Permutes the dimensions of the input according to a given pattern to (time, channel, freq)
    # Useful for e.g. connecting RNNs and convnets together.

    # RNN
    # target_shape: Tuple of integers. Does not include the batch axis.
    # Output shape: (batch_size,) + target_shape
    # target_shape is (the original time dimension, whatever needed to match the dimensions)
    # the latter means concatenate the different activation maps to go:
    # -from a tensor of depth 128 activation maps of time,freq
    # -to a matrix of (time, 128*freq)
    # spec_x = Reshape((data_in.shape[-2], -1))(spec_x)

    spec_rnn = Reshape((params_extract.get('patch_len'), -1))(spec_cnn)
    # for nb_rnn_filt in params_crnn.get('rnn_nb'):
    nb_rnn_filt = params_crnn.get('rnn_nb')
    spec_rnn = Bidirectional(
        GRU(nb_rnn_filt, activation='tanh', dropout=params_crnn.get('dropout_rate'), recurrent_dropout=params_crnn.get('dropout_rate'),
            return_sequences=True),
        merge_mode='mul'
    )(spec_rnn)

    # dims here:
    # -time dim is preserved, untouched
    # -the other dim, 'freq', is 32 due to 32 nodes of the GRU, _r

    # FC - SED, todo esto se podria variar
    sed = spec_rnn
    # for nb_fnn_filt in params_crnn.get('fc_nb'):
    nb_fnn_filt = params_crnn.get('fc_nb')
    sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
    sed = Dropout(params_crnn.get('dropout_rate'))(sed)

    # apply a dense layer to every time frame, mapping it to the number of classes to predict
    # hence we predict class activity on per-frame basis
    # sigmoid for multilabel (will need thresholding in post-pro)

    # originally, we predicted multiple labels for every time slice, hence we can do SED.
    # this is useful for the case where we want to:
    # -determine class activity with timestamps with certain time resolution (hence TimeDistributed)
    # -several classes can overlap (ie multilabel, hence sigmoid & binary_crossentropy)
    # -several classes can occur in the same clip
    # now, the problem is different:
    # -no need for timestamps (hence paso de TimeDistributed)
    # OLD from kaggle18: - no concurrent classes, and only one can occur per file (ie multiclass, hence softmax & categorical_crossentropy)

    # original
    # sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    # sed = Activation('sigmoid', name='sed_out')(sed)

    # we have 2D, need flatten
    # before FC, todo esto se podria variar
    spec_x = Flatten()(sed)

    spec_x = Dense(n_class)(spec_x)
    out = Activation('softmax')(spec_x)

    _model = Model(inputs=spec_start, outputs=out)
    return _model


def get_model_vgg_md(params_learn=None, params_extract=None):
    """

    adapted from
    https://github.com/CPJKU/dcase_task2/blob/master/dcase_task2/models/vgg_gap_spec2.py

    13M params
    """

    K.set_image_data_format('channels_first')
    # n_filt = 64
    # for small datasets
    n_filt = 32

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    channel_axis = 1
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # first conv block
    spec_x = Conv2D(n_filt, (5, 5),
                    strides=2,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Conv2D(n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(spec_x)
    # if params_learn.get('dropout'):
    #     spec_x = Dropout(0.5)(spec_x)
    # else:
    spec_x = Dropout(0.3)(spec_x)

    # second conv block
    spec_x = Conv2D(2*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Conv2D(2*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)
    else:
        spec_x = Dropout(0.3)(spec_x)


    # third conv block
    spec_x = Conv2D(4*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)
    else:
        spec_x = Dropout(0.3)(spec_x)
    spec_x = Conv2D(4*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)
    else:
        spec_x = Dropout(0.3)(spec_x)
    spec_x = Conv2D(6*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)
    else:
        spec_x = Dropout(0.3)(spec_x)
    spec_x = Conv2D(6*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)
    else:
        spec_x = Dropout(0.3)(spec_x)

    # 4th conv block, beware different max pooling operation
    spec_x = Conv2D(8*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Conv2D(8*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = MaxPooling2D(pool_size=(1, 2), data_format="channels_first")(spec_x)
    if params_learn.get('dropout'):
        spec_x = Dropout(0.5)(spec_x)
    else:
        spec_x = Dropout(0.3)(spec_x)

    # 5th conv block, beware different max pooling operation
    spec_x = Conv2D(8*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Dropout(0.5)(spec_x)
    spec_x = Conv2D(8*n_filt, (3, 3),
                    strides=1,
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    # spec_x = MaxPooling2D(pool_size=(1, 2), data_format="channels_first")(spec_x)
    spec_x = Dropout(0.5)(spec_x)

    # 6th conv block, beware dropout and filter size
    # spec_x = Conv2D(8*n_filt, (3, 3),
    #                 strides=1,
    #                 padding='valid',
    #                 kernel_initializer='he_normal',
    #                 data_format='channels_first',
    #                 activation='relu')(spec_x)
    # spec_x = BatchNormalization(axis=1)(spec_x)
    # spec_x = Dropout(0.5)(spec_x)
    spec_x = Conv2D(8*n_filt, (1, 1),
                    strides=1,
                    padding='valid',
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Dropout(0.5)(spec_x)


    # --- feed forward part ---
    spec_x = Conv2D(n_class, (1, 1),
                    kernel_initializer='he_normal',
                    data_format='channels_first',
                    activation='relu')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = GlobalAveragePooling2D(data_format='channels_first')(spec_x)
    # spec_x = Flatten()(spec_x) no need to flatten as the dim is already (batch,41)
    # global average pooling over 41 feature maps (one for each class) followed by a softmax activation.
    # ie global average pooling over per-class feature maps

    # To perform multilabel classification (where each sample can have several classes):
    # end stack of layers with a Dense layer with a number of units equal to the number of classes
    # use sigmoid activation. Replace softmax activation at the end of network with a sigmoid activation
    # use binary_crossentropy loss
    # targets should be k-hot encoded

    out = Dense(n_class,
                kernel_initializer='he_normal',
                # activation='sigmoid',
                activation='softmax',
                name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out)

    return model


# ==================================cochlear.ai================================================
def Conv2D_TD_BAC_mel(filter_size):
    def f(inputs):
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        x = Conv2D(filter_size, (1, 1), padding='same', activation='linear')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filter_size, (3, 3), padding='same', activation='linear')(x)

        se = GlobalAveragePooling2D()(x)
        se = Reshape((1, 1, filter_size))(se)
        se = Dense(filter_size // 2, activation='relu')(se)
        se = Dense(filter_size, activation='sigmoid')(se)
        x = Multiply()([x, se])

        x = Concatenate()([x, inputs])
        return x

    return f


def get_model_cochlear_18(params_learn=None, params_extract=None):

    # def build_model_mel(input_shape, n_class):
    """

    adapted from
    https://github.com/finejuly/dcase2018_task2_cochlearai/blob/master/utils/model.py

    """

    # # build model...
    # print 'build model...'
    #
    # sr = 32000
    # n_fft=1024
    # n_hop=n_fft/8
    #
    # num_class = n_class
    # n_feature = 64
    #
    # inputs = Input(shape=(None,))
    # inputs2 = Reshape((-1,1))(inputs)
    # inputs2 = BatchNormalization()(inputs2)
    # inputs2 = Reshape((1,-1))(inputs2)
    #
    # melspec=Melspectrogram(n_dft=n_fft, n_hop=n_hop, input_shape=(1,None),
    # 						 padding='same', sr=sr, n_mels=n_feature,
    # 						 fmin=0.0, fmax=sr/2, power_melgram=1.0,
    # 						 return_decibel_melgram=True, trainable_fb=False,
    # 						 trainable_kernel=False,
    # 						 name='trainable_stft')(inputs2)

    K.set_image_data_format('channels_last')
    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))

    channel_axis = 3
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # -----low-level K-module***********************************************
    spec_x = BatchNormalization(axis=1)(spec_x)
    # vip reshaped to a size (time, frequency, 1)
    spec_x = Permute((2, 3, 1))(spec_x)  # first conv block
    # x = melspec #(64,none,1)
    # x = Permute((3,2,1))(x) #(1,none,64)
    # x = BatchNormalization()(x)
    # x = Permute((3,2,1    ))(x)

    spec_x1 = Conv2D(15, (3, 3), padding='same', activation='linear')(spec_x)
    spec_x = Concatenate()([spec_x, spec_x1])


    # DenseNet-k-module*****************************************************
    # 8 modules
    for i in range(8):
        spec_x = Conv2D_TD_BAC_mel(np.minimum(512, 16*(2**i)))(spec_x)
        spec_x = MaxPooling2D(pool_size=(2, 2), padding='same')(spec_x)
        # spec_x = Dropout(0.5)(spec_x)

    spec_x = BatchNormalization()(spec_x)
    spec_x = Activation('relu')(spec_x)

    # n-head classifier module**********************************************
    # especially for mixup, we might omit it
    outputsx = [0,0,0,0,0,0,0,0]
    for i in range(8):
        outputs = spec_x
        outputs = Dense(n_class, activation='linear')(outputs)
        outputs = GlobalAveragePooling2D()(outputs)
        outputsx[i] = Activation('softmax')(outputs)
        # outputsx[i] = Activation('sigmoid')(outputs)
        # vip trying with relu for multi-label setting wit sparse labels
        # outputsx[i] = Activation('relu')(outputs)
    outputsx = Average()([o for o in outputsx])
    outputs = outputsx

    # model = Model(inputs=[inputs], outputs=[outputs])
    model = Model(inputs=spec_start, outputs=outputs)

    return model


def get_mobilenet_ka19(params_learn=None, params_extract=None):
    """MobileNet-style CNN, without the classifier layer.
    https://github.com/DCASE-REPO/dcase2019_task2_baseline/blob/master/model.py#L32

    esta net es una traduccion del modelo de Manoj, desde TF slim a keras, pero debe haber algun error
    """
    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    # channel_axis = 1
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # weights_initializer = slim.initializers.xavier_initializer()
    # default uniform=True
    # applies to slim.conv2d, slim.separable_conv2d, slim.fully_connected

    # arg_scope for the  separable_convs
    # with slim.arg_scope([slim.separable_conv2d],
    #                     kernel_size=3, depth_multiplier=1, padding='SAME')



    # net = slim.conv2d(net, 16, kernel_size=3, stride=1, padding='SAME')
    spec_x = Conv2D(16, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='glorot_uniform',
                    data_format='channels_first')(spec_x)

    # net = slim.separable_conv2d(net, 32, stride=2)
    # net = slim.separable_conv2d(net, 32, stride=1)
    # keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    spec_x = SeparableConv2D(32,
                             kernel_size=(3, 3),
                             strides=2,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)
    spec_x = SeparableConv2D(32,
                             kernel_size=(3, 3),
                             strides=1,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)

    # net = slim.separable_conv2d(net, 64, stride=2)
    # net = slim.separable_conv2d(net, 64, stride=1)
    spec_x = SeparableConv2D(64,
                             kernel_size=(3, 3),
                             strides=2,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)
    spec_x = SeparableConv2D(64,
                             kernel_size=(3, 3),
                             strides=1,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)

    # net = slim.separable_conv2d(net, 128, stride=2)
    # net = slim.separable_conv2d(net, 128, stride=1)
    spec_x = SeparableConv2D(128,
                             kernel_size=(3, 3),
                             strides=2,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)
    spec_x = SeparableConv2D(128,
                             kernel_size=(3, 3),
                             strides=1,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)

    # net = slim.separable_conv2d(net, 256, stride=2)
    # net = slim.separable_conv2d(net, 256, stride=1)
    # net = slim.separable_conv2d(net, 256, stride=1)
    spec_x = SeparableConv2D(256,
                             kernel_size=(3, 3),
                             strides=2,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)
    spec_x = SeparableConv2D(256,
                             kernel_size=(3, 3),
                             strides=1,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)
    spec_x = SeparableConv2D(256,
                             kernel_size=(3, 3),
                             strides=1,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)
    # net = slim.separable_conv2d(net, 512, stride=2)
    spec_x = SeparableConv2D(512,
                             kernel_size=(3, 3),
                             strides=2,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)

    spec_x = SeparableConv2D(512,
                             kernel_size=(3, 3),
                             strides=1,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)

    spec_x = SeparableConv2D(512,
                             kernel_size=(3, 3),
                             strides=1,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)

    spec_x = SeparableConv2D(1024,
                             kernel_size=(3, 3),
                             strides=2,
                             padding='same',
                             depth_multiplier=1,
                             kernel_initializer='glorot_uniform',
                             data_format='channels_first'
                             )(spec_x)

    # net = slim.max_pool2d(net, kernel_size=[1,2], stride=1, padding='VALID')
    # net = slim.flatten(net)
    # vip assuming Manoj uses time, freq convention, so preserve time
    spec_x = MaxPooling2D(pool_size=(1, 2),
                          strides=1,
                          padding='valid',
                          data_format="channels_first")(spec_x)

    spec_x = Flatten()(spec_x)

    # Add the logits and the classifier layer.
    # logits = slim.fully_connected(embedding, num_classes, activation_fn=None)
    # prediction = tf.nn.sigmoid(logits)

    # To perform multilabel classification (where each sample can have several classes):
    # end stack of layers with a Dense layer with a number of units equal to the number of classes
    # use sigmoid activation. Replace softmax activation at the end of network with a sigmoid activation
    # use binary_crossentropy loss
    # targets should be k-hot encoded
    out = Dense(n_class,
                kernel_initializer='glorot_uniform',
                activation='sigmoid',
                name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out)

    return model