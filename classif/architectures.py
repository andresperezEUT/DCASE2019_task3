
from keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Activation, Bidirectional, GRU, Reshape, Permute
import keras.backend as K
from keras.models import Model


def get_model_crnn_seld_tagger(params_crnn=None, params_learn=None, params_extract=None):

    K.set_image_data_format('channels_first')

    n_class = params_learn.get('n_classes')

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))

    spec_start = Input(shape=input_shape)
    spec_cnn = spec_start

    for i, convCnt in enumerate(params_crnn.get('cnn_pool_size')):
        spec_cnn = Conv2D(filters=params_crnn.get('cnn_nb_filt'), kernel_size=(3, 3), padding='same')(spec_cnn)
        # spec_cnn = Conv2D(filters=params_crnn.get('cnn_nb_filt'), kernel_size=params_crnn.get('cnn_nb_kernelsize'), padding='same')(spec_cnn)
        # spec_cnn = BatchNormalization(axis=1)(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, params_crnn.get('cnn_pool_size')[i]))(spec_cnn)
        spec_cnn = Dropout(params_crnn.get('dropout_rate'))(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    spec_rnn = Reshape((params_extract.get('patch_len'), -1))(spec_cnn)

    for i, nb_rnn_filt in enumerate(params_crnn.get('rnn_nb')):
        if len(params_crnn.get('rnn_nb')) == 1 or len(params_crnn.get('rnn_nb')) == 2 and i == 1:
            spec_rnn = Bidirectional(
                GRU(nb_rnn_filt, activation='tanh', dropout=params_crnn.get('dropout_rate'), recurrent_dropout=params_crnn.get('dropout_rate'),
                    return_sequences=False), merge_mode='mul'
            )(spec_rnn)

        else:
            spec_rnn = Bidirectional(
                GRU(nb_rnn_filt, activation='tanh', dropout=params_crnn.get('dropout_rate'), recurrent_dropout=params_crnn.get('dropout_rate'),
                    return_sequences=True), merge_mode='mul'
            )(spec_rnn)

    for nb_fnn_filt in params_crnn.get('fc_nb'):
        spec_rnn = Dense(nb_fnn_filt)(spec_rnn)
        spec_rnn = Dropout(params_crnn.get('dropout_rate'))(spec_rnn)

    spec_x = Dense(n_class)(spec_rnn)
    out = Activation('softmax')(spec_x)

    _model = Model(inputs=spec_start, outputs=out)
    return _model
