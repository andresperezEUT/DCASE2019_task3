
import numpy as np
import os, re
import pickle
import json
# import matplotlib.pyplot as plt

from keras.callbacks import Callback
import keras.backend as K
# from keras.callbacks import LearningRateScheduler

# enable to make and save plots in server
# https://github.com/matplotlib/matplotlib/issues/3466/
# option a)
"""uncomment when using GPU servers"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# option b)
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')


"""Basic functions to manage data operations"""

"""
# Some of the functions in this file are taken from [1] and modified
#
# [1] https://github.com/nkundiushuti/pydata2017bcn/blob/master/util.py

"""


def save_tensor(var, out_path=None, suffix='_mel'):
    """
    Saves a numpy array as a binary file
    -review the shape saving when it is a label
    """
    assert os.path.isdir(os.path.dirname(out_path)), "path to save tensor does not exist"
    var.tofile(out_path.replace('.data', suffix + '.data'))
    # save shapes only if not a label
    # todo, shape is not needed for labels, but further changes must be done in other poitns to do it
    # if suffix != '_label':
    save_shape(out_path.replace('.data', suffix + '.shape'), var.shape)


def load_tensor(in_path, suffix=''):
    """
    Loads a binary .data file
    """
    assert os.path.isdir(os.path.dirname(in_path)), "path to load tensor does not exist"
    f_in = np.fromfile(in_path.replace('.data', suffix + '.data'))
    shape = get_shape(in_path.replace('.data', suffix + '.shape'))
    f_in = f_in.reshape(shape)
    return f_in


def save_shape(shape_file, shape):
    """
    Saves the shape of a numpy array
    """
    with open(shape_file, 'w') as fout:
        fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')


def get_shape(shape_file):
    """
    Reads a .shape file
    """
    with open(shape_file, 'rb') as f:
        line=f.readline().decode('ascii')
        if line.startswith('#'):
            shape=tuple(map(int, re.findall(r'(\d+)', line)))
            return shape
        else:
            raise IOError('Failed to find shape in file')


def chunker(seq, size):
    """return one batch of given size within a list of items in seq"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def save_var2disk(var, out_path=None, save=True, suffix='_mel_'):
    if save and out_path is not None:
        save_tensor(var, out_path, suffix)
        var = None
    else:
        return var


def get_num_instances_per_file(f_name, patch_len=25, patch_hop=12):
    """
    Return the number of context_windows or instances generated out of a given file
    """
    shape = get_shape(os.path.join(f_name.replace('.data', '.shape')))
    file_frames = float(shape[0])
    return np.maximum(1, int(np.ceil((file_frames-patch_len)/patch_hop)))


def get_feature_size_per_file(f_name):
    """
    Return the dimensionality of the features in a given file.
    Typically, this will be the number of bins in a T-F representation
    """
    shape = get_shape(os.path.join(f_name.replace('.data', '.shape')))
    return shape[1]


def make_sure_isdir(pre_path, _out_file):
    """
    make sure the a directory at the end of pre_path exists. Else create it
    :param pre_path:
    :param args:
    :return:
    """
    full_path = os.path.join(pre_path, _out_file)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


def save_learning_curves(params_ctrl=None, history=None):
    """
    save learning curves for acc and loss
    we check for existing files and add vx

    :param params_ctrl:
    :param history:
    :return:
    """
    path_pics = make_sure_isdir('logs/pics', params_ctrl.get('output_file'))
    fig_name = params_ctrl.get('output_file') + '_v' + str(params_ctrl.get('count_trial'))

    if isinstance(history, dict):
        # I passed the dict directly

        # we have the proper fig_name (to be complemented with acc or loss
        # summarize history for accuracy
        plt.figure()
        plt.plot(history['acc'], "b.-")
        plt.plot(history['val_acc'], "g.-")
        plt.axis([-2, 101, 0.15, 0.95])
        plt.title('learning curves - accuracy')
        plt.ylabel('accuracy [%]')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(path_pics, fig_name + '_acc_.png'), bbox_inches='tight')
        plt.close()

        # summarize history for loss
        plt.figure()
        plt.plot(history['loss'], "b.-")
        plt.plot(history['val_loss'], "g.-")
        plt.axis([-2, 101, 0.05, 3.0])
        plt.title('learning curves - loss')
        plt.ylabel('loss [ ]')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(path_pics, fig_name + '_loss_.png'), bbox_inches='tight')
        plt.close()

        # summarize history for lr
        plt.figure()
        plt.plot(history['lr'], "r.-")
        plt.axis([-2, 101, 0, 0.001])
        plt.title('learning curves - lr')
        plt.ylabel('lr [ ]')
        plt.xlabel('epoch')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(path_pics, fig_name + '_lr_.png'), bbox_inches='tight')
        plt.close()

    else:
        # I passed the full keras history object

        # we have the proper fig_name (to be complemented with acc or loss
        # summarize history for accuracy
        plt.figure()
        plt.plot(history.history['acc'], "b.-")
        plt.plot(history.history['val_acc'], "g.-")
        plt.axis([-2, 101, 0.15, 0.95])
        plt.title('learning curves - accuracy')
        plt.ylabel('accuracy [%]')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(path_pics, fig_name + '_acc_.png'), bbox_inches='tight')
        plt.close()

        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'], "b.-")
        plt.plot(history.history['val_loss'], "g.-")
        plt.axis([-2, 101, 0.05, 3.0])
        plt.title('learning curves - loss')
        plt.ylabel('loss [ ]')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(path_pics, fig_name + '_loss_.png'), bbox_inches='tight')
        plt.close()

        # summarize history for lr
        plt.figure()
        plt.plot(history.history['lr'], "r.-")
        plt.axis([-2, 101, 0, 0.001])
        plt.title('learning curves - lr')
        plt.ylabel('lr [ ]')
        plt.xlabel('epoch')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(path_pics, fig_name + '_lr_.png'), bbox_inches='tight')
        plt.close()

    print('\nSuccessfully saved learning curves for: %s' % fig_name)
    return 0


def save_history(params_ctrl=None, history=None):
    """
    :param params_ctrl:
    :param history:
    :return:
    """

    path_pics = make_sure_isdir('logs/pics', params_ctrl.get('output_file'))
    fig_name = params_ctrl.get('output_file') + '_v' + str(params_ctrl.get('count_trial'))

    history_filename = os.path.join(path_pics, fig_name + '.pickle')
    pickle.dump(history, open(history_filename, "wb"))
    # val_loss = hist.history.get('val_loss')[-1]
    # tr_loss = hist.history.get('loss')[-1]
    print('\nSuccessfully saved pickle history for: %s' % fig_name)
    return 0


def save_softmax(params_ctrl=None, values=None):
    """
    :param params_ctrl:
    :param history:
    :return:
    """
    # path includes folder with experiment name
    path_pics = make_sure_isdir('logs/pics', params_ctrl.get('output_file'))
    # file_name includes experiment name + softmax_v18
    file_name = params_ctrl.get('output_file') + '_softmax_v' + str(params_ctrl.get('count_trial'))

    softmax_filename = os.path.join(path_pics, file_name + '.pickle')
    pickle.dump(values, open(softmax_filename, "wb"))
    # val_loss = hist.history.get('val_loss')[-1]
    # tr_loss = hist.history.get('loss')[-1]
    print('Successfully saved pickle with softmax values for: %s' % file_name)
    return 0


class GetCurrentEpoch(Callback):
    """get the current epoch to pass it within the loss function.
    # Arguments
        current_epoch: The tensor withholding the current epoch on_epoch_begin
    """

    def __init__(self, current_epoch):
        super(GetCurrentEpoch, self).__init__()
        self.current_epoch = current_epoch

    def on_epoch_begin(self, epoch, logs=None):
        new_epoch = epoch
        # Set new value
        K.set_value(self.current_epoch, new_epoch)
