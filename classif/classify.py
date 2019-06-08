
import os

# basic
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm, trange
import time
import pprint
import datetime
import argparse
from scipy.stats import gmean
import yaml
import shutil
import pickle

# keras
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard

# DIY
import utils_classif

# import matplotlib.pyplot as plt, done in utils_classif
from feat_ext import load_audio_file, get_mel_spectrogram, modify_file_variable_length
from data import get_label_files, DataGeneratorPatch, PatchGeneratorPerFile
from architectures import build_model_tf_basic, get_model_tf_js, \
    get_model_crnn_sa, get_model_tf_js_tidy, get_model_vgg_md, get_model_cochlear_18, get_model_tf_kong_cnnbase18, \
    get_mobilenet_ka19, get_model_crnn_seld, get_model_crnn_seld_tagger
from eval import Evaluator
from losses import crossentropy_cochlear, crossentropy_diy_max, lq_loss, lq_loss_wrap, crossentropy_diy_max_wrap, \
    crossentropy_diy_outlier_wrap, crossentropy_reed_wrap, crossentropy_diy_outlier_origin_wrap
from mobilenet import mobilenet

# andres-----------------------------------------
# from parameters import get_params
import csv
import sys
# wanna import functions from modules in the parent directory
sys.path.append('../')
from parameters import get_params
from compute_doa_metrics import compute_DOA_metrics
from file_utils import write_metadata_result_file, build_result_dict_from_metadata_array, write_output_result_file


start = time.time()

now = datetime.datetime.now()
print("Current date and time:")
print(str(now))

# =========================================================================================================
# =========================================================================================================

# ==================================================================== ARGUMENTS
parser = argparse.ArgumentParser(description='Experiments on audio with noisy labels')
parser.add_argument('-p', '--params_yaml',
                    dest='params_yaml',
                    action='store',
                    required=False,
                    type=str)
args = parser.parse_args()
print('\nYaml file with parameters defining the experiment: %s\n' % str(args.params_yaml))


# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables

# Read parameters file from yaml passed by argument
params = yaml.load(open(args.params_yaml))
params_ctrl = params['ctrl']
params_extract = params['extract']
params_learn = params['learn']
params_loss = params['loss']
params_recog = params['recognizer']
params_crnn = params['crnn']

suffix_in = params['suffix'].get('in')
suffix_out = params['suffix'].get('out')

# overwrite preactivation for these 2 models
if params_learn.get('model') == 'js_tidy':
    params_learn['preactivation'] = 4
elif params_learn.get('model') == 'kong':
    params_learn['preactivation'] = 0
else:
    print('model selection is not tailored to a preactivation approach')

# determine loss function for stage 1 (or entire training)
flag_origin = False
if params_loss.get('type') == 'CCE':
    params_loss['type'] = 'categorical_crossentropy'
elif params_loss.get('type') == 'MAE':
    params_loss['type'] = 'mean_absolute_error'
elif params_loss.get('type') == 'lq_loss':
    params_loss['type'] = lq_loss_wrap(params_loss.get('q_loss'))
elif params_loss.get('type') == 'CCE_diy_max':
    params_loss['type'] = crossentropy_diy_max_wrap(params_loss.get('m_loss'))
elif params_loss.get('type') == 'CCE_diy_outlier':
    params_loss['type'] = crossentropy_diy_outlier_wrap(params_loss.get('l_loss'))
elif params_loss.get('type') == 'CCE_reed':
    params_loss['type'] = crossentropy_reed_wrap(params_loss.get('reed_type'), params_loss.get('reed_beta'))

params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))
#

#  vip to deploy. for public, put directly params_ctrl.gt('dataset_path') within params_path
path_root_data = params_ctrl.get('dataset_path')

# watch remove this if for public ======================================================== PATHS FOR DATA, FEATURES and GROUND TRUTH
# vip this if statement determines where to look for the dataset
if params_ctrl.get('execute') == 'gpu_ds':
    # path_root_data = os.path.join('/data/FSDKaggle2018x', params_ctrl.get('proto'))
    print('path_root_data taken from yaml')
elif params_ctrl.get('execute') == 'cpu':
    print('path_root_data taken from yaml')

params_path = {'path_to_features': os.path.join(path_root_data, 'features'),
               # 'featuredir_dev': 'audio_dev_varup1/',
               # 'featuredir_eval': 'audio_eval_varup1/',
               'featuredir_dev': 'audio_dev_varup2_64mel/',
               'featuredir_eval': 'audio_eval_varup2_64mel/',
               # 'featuredir_dev_param': 'audio_dev_param_varup2_64mel/',
               # 'featuredir_eval_param': 'audio_eval_param_varup2_64mel/',
               'featuredir_dev_param': 'audio_dev_param_Q_varup2_64mel/',
               'featuredir_eval_param': 'audio_eval_param_Q_varup2_64mel/',
               # 'featuredir_dev': 'audio_dev_varup1_64mel/',
               # 'featuredir_eval': 'audio_eval_varup1_64mel/',
               'path_to_dataset': path_root_data,
               'audiodir_dev': 'wav/dev/',
               'audiodir_eval': 'wav/eval/',
               # 'audiodir_dev_param': 'wav/dev_param/',
               # 'audiodir_eval_param': 'wav/eval_param/',
               'audiodir_dev_param': 'wav/dev_param_Q/',
               'audiodir_eval_param': 'wav/eval_param_Q/',
               'audio_shapedir_dev': 'audio_dev_shapes/',
               'audio_shapedir_eval': 'audio_eval_shapes/',
               # 'audio_shapedir_dev_param': 'audio_dev_param_shapes/',
               # 'audio_shapedir_eval_param': 'audio_eval_param_shapes/',
               'audio_shapedir_dev_param': 'audio_dev_param_Q_shapes/',
               'audio_shapedir_eval_param': 'audio_eval_param_Q_shapes/',
               'gt_files': path_root_data}

if params_extract.get('n_mels') == 40:
    params_path['featuredir_dev'] = 'audio_dev_varup2_40mel/'
    params_path['featuredir_eval'] = 'audio_eval_varup2_40mel/'
    # params_path['featuredir_dev_param'] = 'audio_dev_param_varup2_40mel/'
    # params_path['featuredir_eval_param'] = 'audio_eval_param_varup2_40mel/'
    params_path['featuredir_dev_param'] = 'audio_dev_param_Q_varup2_40mel/'
    params_path['featuredir_eval_param'] = 'audio_eval_param_Q_varup2_40mel/'
elif params_extract.get('n_mels') == 96:
    params_path['featuredir_dev'] = 'audio_dev_varup2_96mel/'
    params_path['featuredir_eval'] = 'audio_eval_varup2_96mel/'
    # params_path['featuredir_dev_param'] = 'audio_dev_param_varup2_96mel/'
    # params_path['featuredir_eval_param'] = 'audio_eval_param_varup2_96mel/'
    params_path['featuredir_dev_param'] = 'audio_dev_param_Q_varup2_96mel/'
    params_path['featuredir_eval_param'] = 'audio_eval_param_Q_varup2_96mel/'
elif params_extract.get('n_mels') == 128:
    params_path['featuredir_dev'] = 'audio_dev_varup2_128mel/'
    params_path['featuredir_eval'] = 'audio_eval_varup2_128mel/'
    # params_path['featuredir_dev_param'] = 'audio_dev_param_varup2_128mel/'
    # params_path['featuredir_eval_param'] = 'audio_eval_param_varup2_128mel/'
    params_path['featuredir_dev_param'] = 'audio_dev_param_Q_varup2_128mel/'
    params_path['featuredir_eval_param'] = 'audio_eval_param_Q_varup2_128mel/'


if params_learn.get('mixup_log'):
    # we will be computing log of spectrograms on the fly for training (after mixup)
    # for the test set, the log is applied pre-computed
    # here we define a new folder for the train features, where NO log is applied
    params_path['featuredir_dev'] = 'audio_dev_varup2_nolog/'
    print('\n===Using dev set WITHOUT logarithm on the spectrograms.')
    # since this folder is non existing, it will compute features, with the log disabled in the feat extraction

params_path['featurepath_dev'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_dev'))
params_path['featurepath_eval'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_eval'))
params_path['featurepath_dev_param'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_dev_param'))
params_path['featurepath_eval_param'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_eval_param'))

params_path['audiopath_dev'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_dev'))
params_path['audiopath_eval'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_eval'))
params_path['audiopath_dev_param'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_dev_param'))
params_path['audiopath_eval_param'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_eval_param'))


params_path['audio_shapedir_dev'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_dev'))
params_path['audio_shapedir_eval'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_eval'))
params_path['audio_shapedir_dev_param'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_dev_param'))
params_path['audio_shapedir_eval_param'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_eval_param'))


# ======================================================== SPECIFIC PATHS TO SOME IMPORTANT FILES
# ground truth, load model, save model, predictions, results
params_files = {'gt_eval': os.path.join(params_path.get('gt_files'), 'gt_eval.csv'),
                'gt_dev': os.path.join(params_path.get('gt_files'), 'gt_dev.csv'),
                'load_model': 'trained_models/T_F_lee_varup_lr0.001.h5'}

path_trained_models = utils_classif.make_sure_isdir('trained_models', params_ctrl.get('output_file'))
params_files['save_model'] = os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' + str(params_ctrl.get('count_trial')) + '.h5')
path_predictions = utils_classif.make_sure_isdir('predictions', params_ctrl.get('output_file'))
params_files['predictions'] = os.path.join(path_predictions, params_ctrl.get('output_file') + '_v' + str(params_ctrl.get('count_trial')) + '.csv')
path_results = utils_classif.make_sure_isdir('logs/results', params_ctrl.get('output_file'))
params_files['results'] = os.path.join(path_results, params_ctrl.get('output_file') + '.pickle')
# params_files['event_durations'] = os.path.join('logs/pics', params_ctrl.get('output_file') + '_event_durations.pickle')

# # ============================================= print all params to keep record in output file
print('\nparams_ctrl=')
pprint.pprint(params_ctrl, width=1, indent=4)
print('params_files=')
pprint.pprint(params_files, width=1, indent=4)
print('params_extract=')
pprint.pprint(params_extract, width=1, indent=4)
print('params_learn=')
pprint.pprint(params_learn, width=1, indent=4)
print('params_loss=')
pprint.pprint(params_loss, width=1, indent=4)
print('params_recog=')
pprint.pprint(params_recog, width=1, indent=4)
print('params_crnn=')
pprint.pprint(params_crnn, width=1, indent=4)
print('\n')


# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA

# aim: lists with all wav files for dev, which includes train/val/test
gt_dev = pd.read_csv(params_files.get('gt_dev'))
splitlist_audio_dev = gt_dev.split.values.tolist()
# test_csv = pd.read_csv(params_files.get('gt_test')) todo
# the following are lists of wav files (str), eg [12345.wav, 98765.wav, 424000.wav, 105789.wav, ...]
filelist_audio_dev = gt_dev.fname.values.tolist()
# filelist_audio_te = test_csv.fname.values.tolist() todo

# create dict with ground truth mapping with labels:
# -key: path to wav
# -value: the ground truth label too
file_to_label = {params_path.get('audiopath_dev') + k: v for k, v in zip(gt_dev.fname.values, gt_dev.label.values)}

# ========================================================== CREATE VARS FOR DATASET MANAGEMENT
# list with unique n_classes labels and aso_ids
list_labels = sorted(list(set(gt_dev.label.values)))

# create dicts such that key: value is as follows
# fixed by DCASE
label_to_int = {
        'clearthroat': 2,
        'cough': 8,
        'doorslam': 9,
        'drawer': 1,
        'keyboard': 6,
        'keysDrop': 4,
        'knock': 0,
        'laughter': 10,
        'pageturn': 7,
        'phone': 3,
        'speech': 5
}
int_to_label = {v: k for k, v in label_to_int.items()}

# create ground truth mapping with categorical values
file_to_label_numeric = {k: label_to_int[v] for k, v in file_to_label.items()}


#
# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# compute T_F representation
# mel-spectrogram for all files in the dataset and store it
var_lens = {item: [] for item in label_to_int.keys()}
var_lens['overall'] = []

var_lens_dev_param = {}
var_lens_dev_param['overall'] = []

if params_ctrl.get('feat_ext'):
    if params_ctrl.get('pipeline') == 'T_F':
        n_extracted_dev = 0; n_extracted_te = 0; n_failed_dev = 0; n_failed_te = 0
        n_extracted_dev_param = 0; n_failed_dev_param = 0

        # only if features have not been extracted, ie
        # if folder does not exist, or it exists with less than 80% of the feature files
        # create folder and extract features
        nb_files_dev = len(filelist_audio_dev)
        if not os.path.exists(params_path.get('featurepath_dev')) or \
                        len(os.listdir(params_path.get('featurepath_dev'))) < nb_files_dev*0.8:

            if os.path.exists(params_path.get('featurepath_dev')):
                shutil.rmtree(params_path.get('featurepath_dev'))
            os.makedirs(params_path.get('featurepath_dev'))

            print('\nFeature extraction for dev set (prints enabled). Features dumped in {}.........................'.
                  format(params_path.get('featurepath_dev')))
            for idx, f_name in enumerate(filelist_audio_dev):
                f_path = os.path.join(params_path.get('audiopath_dev'), f_name)
                if os.path.isfile(f_path) and f_name.endswith('.wav'):
                    # load entire audio file and modify variable length, if needed
                    y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)

                    # keep record of the lengths, per class, for insight
                    duration_seconds = len(y)/int(params_extract.get('fs'))
                    var_lens[f_name.split('_')[0]].append(duration_seconds)
                    var_lens['overall'].append(duration_seconds)

                    y = modify_file_variable_length(data=y,
                                                    input_fixed_length=params_extract['audio_len_samples'],
                                                    params_extract=params_extract)
                    # print('Considered audio length: %6.3f' % (len(y) / params_extract.get('fs')))
                    # print('%-22s: [%d/%d] of %s' % ('Extracting tr features', (idx + 1), nb_files_tr, f_path))

                    # compute log-scaled mel spec. row x col = time x freq
                    # this is done only for the length specified by loading mode (fix, varup, varfull)
                    mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                    # save the T_F rep to a binary file (only the considered length)
                    utils_classif.save_tensor(var=mel_spectrogram,
                                              out_path=os.path.join(params_path.get('featurepath_dev'),
                                                                    f_name.replace('.wav', '.data')), suffix='_mel')

                    # save also label
                    utils_classif.save_tensor(var=np.array([file_to_label_numeric[f_path]], dtype=float),
                                              out_path=os.path.join(params_path.get('featurepath_dev'),
                                                                    f_name.replace('.wav', '.data')), suffix='_label')

                    if os.path.isfile(os.path.join(params_path.get('featurepath_dev'),
                                                   f_name.replace('.wav', suffix_in + '.data'))):
                        n_extracted_dev += 1
                        print('%-22s: [%d/%d] of %s' % ('Extracted dev features', (idx + 1), nb_files_dev, f_path))
                    else:
                        n_failed_dev += 1
                        print('%-22s: [%d/%d] of %s' % ('FAILING to extract dev features', (idx + 1), nb_files_dev, f_path))
                else:
                    print('%-22s: [%d/%d] of %s' % ('this dev audio is in the csv but not in the folder', (idx + 1), nb_files_dev, f_path))

            print('n_extracted_dev: {0} / {1}'.format(n_extracted_dev, nb_files_dev))
            print('n_failed_dev: {0} / {1}\n'.format(n_failed_dev, nb_files_dev))

        else:
            print('Dev set is already extracted in {}'.format(params_path.get('featurepath_dev')))


        # do feature extraction for dev_param (outcome of Andres approaches)
        audio_files_dev_param = [f for f in os.listdir(params_path.get('audiopath_dev_param')) if not f.startswith('.')]

        nb_files_dev_param = len(audio_files_dev_param)
        if not os.path.exists(params_path.get('featurepath_dev_param')) or \
                        len(os.listdir(params_path.get('featurepath_dev_param'))) < nb_files_dev_param * 0.8:

            if os.path.exists(params_path.get('featurepath_dev_param')):
                shutil.rmtree(params_path.get('featurepath_dev_param'))
            os.makedirs(params_path.get('featurepath_dev_param'))

            print(
                '\nFeature extraction for dev set parametric (outcome of Andres). Features dumped in {}.........................'.
                format(params_path.get('featurepath_dev_param')))
            for idx, f_name in enumerate(audio_files_dev_param):
                f_path = os.path.join(params_path.get('audiopath_dev_param'), f_name)
                if os.path.isfile(f_path) and f_name.endswith('.wav'):
                    # load entire audio file and modify variable length, if needed
                    y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'],
                                        params_extract=params_extract)

                    # keep record of the lengths, per class, for insight
                    duration_seconds = len(y) / int(params_extract.get('fs'))
                    var_lens_dev_param['overall'].append(duration_seconds)

                    y = modify_file_variable_length(data=y,
                                                    input_fixed_length=params_extract['audio_len_samples'],
                                                    params_extract=params_extract)
                    # print('Considered audio length: %6.3f' % (len(y) / params_extract.get('fs')))
                    # print('%-22s: [%d/%d] of %s' % ('Extracting tr features', (idx + 1), nb_files_tr, f_path))

                    # compute log-scaled mel spec. row x col = time x freq
                    # this is done only for the length specified by loading mode (fix, varup, varfull)
                    mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                    # save the T_F rep to a binary file (only the considered length)
                    utils_classif.save_tensor(var=mel_spectrogram,
                                              out_path=os.path.join(params_path.get('featurepath_dev_param'),
                                                                    f_name.replace('.wav', '.data')), suffix='_mel')

                    # save also label. NO
                    # utils_classif.save_tensor(var=np.array([file_to_label_numeric[f_path]], dtype=float),
                    #                           out_path=os.path.join(params_path.get('featurepath_dev_param'),
                    #                                                 f_name.replace('.wav', '.data')),
                    #                           suffix='_label')

                    if os.path.isfile(os.path.join(params_path.get('featurepath_dev_param'),
                                                   f_name.replace('.wav', suffix_in + '.data'))):
                        n_extracted_dev_param += 1
                        print('%-22s: [%d/%d] of %s' % ('Extracted dev_param features', (idx + 1), nb_files_dev_param, f_path))
                    else:
                        n_failed_dev_param += 1
                        print('%-22s: [%d/%d] of %s' % (
                        'FAILING to extract dev_param features', (idx + 1), nb_files_dev_param, f_path))
                else:
                    print('%-22s: [%d/%d] of %s' % (
                    'this dev_param audio is in the csv but not in the folder', (idx + 1), nb_files_dev_param, f_path))

            print('n_extracted_dev_param: {0} / {1}'.format(n_extracted_dev_param, nb_files_dev_param))
            print('n_failed_dev_param: {0} / {1}\n'.format(n_failed_dev_param, nb_files_dev_param))

        else:
            print('Dev_param set is already extracted in {}'.format(params_path.get('featurepath_dev_param')))


        # save dict with event durations, algo cambie aqui y lie los paths
        # path_pics = utils_classif.make_sure_isdir('logs/pics', params_ctrl.get('output_file'))
        # pickle.dump(var_lens, open(path_pics + '/event_durations.pickle'), 'wb')

        # todo when eval set is available******************
        # if not os.path.exists(params_path.get('featurepath_te')):
        #     os.makedirs(params_path.get('featurepath_te'))
        #
        #     print('\nFeature extraction for test set (prints disabled)............................................')
        #
        #     nb_files_te = len(filelist_audio_te)
        #     for idx, f_name in enumerate(filelist_audio_te):
        #         f_path = os.path.join(params_path.get('audiopath_te'), f_name)
        #         if os.path.isfile(f_path) and f_name.endswith('.wav'):
        #             # load entire audio file and modify variable length, if needed
        #             y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
        #             y = modify_file_variable_length(data=y,
        #                                             input_fixed_length=params_extract['audio_len_samples'],
        #                                             params_extract=params_extract)
        #             # print('Considered audio length: %6.3f' % (len(y) / params_extract.get('fs')))
        #             # print('%-22s: [%d/%d] of %s' % ('Extracting te features', (idx + 1), nb_files_te, f_path))
        #
        #             # compute log-scaled mel spec. row x col = time x freq
        #             # this is done only for the length specified by loading mode (fix, varup, varfull)
        #             mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)
        #
        #             # save the T_F rep to a binary file (only the considered length)
        #             utils_classif.save_tensor(var=mel_spectrogram,
        #                               out_path=os.path.join(params_path.get('featurepath_te'),
        #                                                       f_name.replace('.wav', '.data')), suffix='_mel')
        #
        #             # save also labels. we dont have the ground truth for test set at this point
        #             # save_var2disk(var=np.array([file_to_label_numeric[f_path]], dtype=float),
        #             #               out_path=os.path.join(path_to_features, 'audio_test/', f.replace('.wav', '.data')),
        #             #               save=True,
        #             #               suffix='_label')
        #
        #             if os.path.isfile(os.path.join(params_path.get('featurepath_te'),
        #                                            f_name.replace('.wav', '_mel.data'))):
        #                 n_extracted_te += 1
        #                 print('%-22s: [%d/%d] of %s' % ('Extracted te features', (idx + 1), nb_files_te, f_path))
        #             else:
        #                 n_failed_te += 1
        #                 print('%-22s: [%d/%d] of %s' % ('FAILING to extract te features', (idx + 1), nb_files_te, f_path))
        #         else:
        #             print('%-22s: [%d/%d] of %s' % ('this te audio is in the csv but not in the folder', (idx + 1), nb_files_te, f_path))
        #
        #     print('n_extracted_te: {0} / {1}'.format(n_extracted_te, nb_files_te))
        #     print('n_failed_te: {0} / {1}\n'.format(n_failed_te, nb_files_te))
        #
        # else:
        #     print('Test set is already extracted in {}'.format(params_path.get('featurepath_te')))
        #
        # # here we have all train set and test in a T_F rep
        # # and stored in path_to_features

##

# Assumes features or T-F representations on a per-file fashion are previously computed and in disk
# list .data files in train set
# input: '_mel', '_gamma', '_cqt'
# output: '_label'

# vip select the subset of training data to consider: all, clean, noisy, noisy_small
# =====================================================================================================================
# =====================================================================================================================

# based on the feature files, in case some wav file could not be converted to features
# only files (not path), feature file list for tr, eg [1234_mel.data, 230987_mel.data, ...]
# ff_list_tr = [f for f in os.listdir(params_path.get('featurepath_tr')) if f.endswith(suffix_in + '.data') and
#               os.path.isfile(os.path.join(params_path.get('featurepath_tr'), f.replace(suffix_in, suffix_out)))]

ff_list_dev = [filelist_audio_dev[i].replace('.wav', suffix_in + '.data') for i in range(len(filelist_audio_dev))]

# get label for every file *from the .data saved in disk*, in float
# labels_audio_train is an ndarray of floats with sizeof(ff_list_tr), ie (num_files, 1) (singly-labeled data)
labels_audio_dev = get_label_files(filelist=ff_list_dev,
                                   dire=params_path.get('featurepath_dev'),
                                   suffix_in=suffix_in,
                                   suffix_out=suffix_out
                                   )

# sanity check
print('Number of clips considered as dev set: {0}'.format(len(ff_list_dev)))
print('Number of labels loaded for dev set: {0}'.format(len(labels_audio_dev)))
scalers = [None]*4
# vip determine the validation setup according to the folds, and perform training / val / test for each fold
for kfo in range(1, 5):
# for kfo in range(1, 2):
    print('\n=========================================================================================================')
    print('===Processing fold {} within the x-val setup...'.format(kfo))
    print('=========================================================================================================\n')
    # x-val setup given by DCASE organizers
    if kfo == 1:
        splits_tr = [3, 4]
        splits_val = [2]
        splits_te = [1]
    elif kfo == 2:
        splits_tr = [4, 1]
        splits_val = [3]
        splits_te = [2]
    elif kfo == 3:
        splits_tr = [1, 2]
        splits_val = [4]
        splits_te = [3]
    elif kfo == 4:
        splits_tr = [2, 3]
        splits_val = [1]
        splits_te = [4]

    params_ctrl['current_fold'] = kfo
    tr_files0 = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_tr[0]]
    tr_files1 = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_tr[1]]
    tr_files = tr_files0 + tr_files1
    val_files = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_val[0]]
    te_files = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_te[0]]

    # SC
    if len(tr_files) + len(val_files) + len(te_files) != len(ff_list_dev):
        print('ERROR: You messed up in x-val setup for fold: {0}'.format(len(kfo)))
        print('{} is not {}'.format(len(tr_files) + len(val_files) + len(te_files), len(ff_list_dev)))

    # ============================================================BATCH GENERATION
    # ============================================================BATCH GENERATION

    tr_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_dev'),
                                      file_list=tr_files,
                                      params_learn=params_learn,
                                      params_extract=params_extract,
                                      suffix_in='_mel',
                                      suffix_out='_label',
                                      floatx=np.float32
                                      )
    # to predict later on on dev_param clips
    scalers[kfo-1] = tr_gen_patch.scaler

    print("Total number of instances *only* for training: %s" % str(tr_gen_patch.nb_inst_total))
    print("Batch_size: %s" % str(tr_gen_patch.batch_size))
    print("Number of iterations (batches) in the training subset: %s" % str(tr_gen_patch.nb_iterations))
    print("\nShape of training subset: %s" % str(tr_gen_patch.features.shape))
    print("Shape of labels in training subset: %s" % str(tr_gen_patch.labels.shape))

    # compute class_weigths based on the labels generated
    if params_learn.get('mode_class_weight'):
        labels_nice = np.reshape(tr_gen_patch.labels, -1)  # remove singleton dimension
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(labels_nice),
                                                          labels_nice)
        class_weights_dict = dict(enumerate(class_weights))
    else:
        class_weights_dict = None

    val_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_dev'),
                                       file_list=val_files,
                                       params_learn=params_learn,
                                       params_extract=params_extract,
                                       suffix_in='_mel',
                                       suffix_out='_label',
                                       floatx=np.float32,
                                       scaler=tr_gen_patch.scaler
                                       )

    print("\nShape of validation subset: %s" % str(val_gen_patch.features.shape))
    print("Shape of labels in validation subset: %s" % str(val_gen_patch.labels.shape))

    # --------------------------------sanity check
    batch_features_tr, batch_labels_tr = tr_gen_patch.__getitem__(0)
    print(batch_features_tr.shape)
    # (batch_size, 1, time, freq)
    print(batch_labels_tr.shape)
    label = np.nonzero(batch_labels_tr[2, :])[0]
    print("Category (nonzero): {}".format([int_to_label[lab] for lab in label]))
    print("Category (max): {}".format(int_to_label[np.argmax(batch_labels_tr[2, :])]))

    batch_features_val, batch_labels_val = val_gen_patch.__getitem__(0)
    print(batch_features_val.shape)
    # (batch_size, 1, time, freq)
    print(batch_labels_val.shape)
    label = np.nonzero(batch_labels_val[2, :])[0]
    print("Category (nonzero): {}".format([int_to_label[lab] for lab in label]))
    print("Category (max): {}".format(int_to_label[np.argmax(batch_labels_val[2, :])]))

    # ============================================================DEFINE AND FIT A MODEL
    # ============================================================DEFINE AND FIT A MODEL

    tr_loss, val_loss = [0] * params_learn.get('n_epochs'), [0] * params_learn.get('n_epochs')
    # ============================================================
    if params_ctrl.get('learn'):
        if params_learn.get('model') == 'debug':
            # load TO DEBUG
            model = build_model_tf_basic(params_learn=params_learn, params_extract=params_extract)

        elif params_learn.get('model') == 'js':
            # load JS
            model = get_model_tf_js(params_learn=params_learn, params_extract=params_extract)

        elif params_learn.get('model') == 'js_tidy':
            # load JS with typical order
            model = get_model_tf_js_tidy(params_learn=params_learn, params_extract=params_extract)

        elif params_learn.get('model') == 'kong':
            # load Kong's baseline CNN for DCASE 2018
            model = get_model_tf_kong_cnnbase18(params_learn=params_learn, params_extract=params_extract)

        elif params_learn.get('model') == 'crnn':
            # load CRNN********************************************************************
            # CRNN model definition
            # cnn_nb_filt = 128  # CNN filter size. originally 128, but in papers, from 8 to 128
            # cnn_pool_size = [5, 2, 2]  # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
            # rnn_nb = [32, 32]  # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
            # fc_nb = [32]  # Number of FC nodes.  Length of fc_nb =  number of FC layers
            # dropout_rate = 0.5  # Dropout after each layer

            model = get_model_crnn_sa(params_crnn=params_crnn, params_learn=params_learn,
                                      params_extract=params_extract)

        elif params_learn.get('model') == 'crnn_seld':
            model = get_model_crnn_seld(params_crnn=params_crnn, params_learn=params_learn,
                                        params_extract=params_extract)

        elif params_learn.get('model') == 'crnn_seld_tagger':
            model = get_model_crnn_seld_tagger(params_crnn=params_crnn, params_learn=params_learn,
                                               params_extract=params_extract)

        elif params_learn.get('model') == 'vgg_md':
            model = get_model_vgg_md(params_learn=params_learn, params_extract=params_extract)

        elif params_learn.get('model') == 'cochlear':
            model = get_model_cochlear_18(params_learn=params_learn, params_extract=params_extract)

        elif params_learn.get('model') == 'mobile':
            # not this one, has errors
            model = get_mobilenet_ka19(params_learn=params_learn, params_extract=params_extract)

        elif params_learn.get('model') == 'mobileKERAS':
            # this one is the original
            model = mobilenet(alpha=params_learn.get('alpha_mobilenet'),
                              depth_multiplier=1,
                              dropout=0.5,
                              include_top=True,
                              weights=None,
                              input_tensor=None,
                              pooling=None,
                              classes=11,
                              params_learn=params_learn,
                              params_extract=params_extract
                              )

        if params_learn.get('stages') == 0:
            # vip implementing a warmup period for mixup*******************************************************

            opt = Adam(lr=params_learn.get('lr'))
            model.compile(optimizer=opt, loss=params_loss.get('type'), metrics=['accuracy'])
            model.summary()

            # watch warmup****************************************** no mixup is applied
            print('\n===implementing a warmup period for mixup: WARMUP STAGE (no mixup)**************************')
            params_learn['mixup'] = False
            tr_gen_patch_mixup_warmup = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                                           file_list=tr_files,
                                                           params_learn=params_learn,
                                                           params_extract=params_extract,
                                                           suffix_in='_mel',
                                                           suffix_out='_label',
                                                           floatx=np.float32
                                                           )

            # callbacks
            reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
            path_tb = 'logs/pics/tb/{}/v{}'.format(params_ctrl.get('output_file'), str(params_ctrl.get('count_trial')))
            tensorboard = TensorBoard(log_dir=path_tb,
                                      histogram_freq=0,
                                      batch_size=params_learn.get('batch_size'),
                                      write_grads=True,
                                      write_images=True,
                                      )
            callback_list = [tensorboard, reduce_lr]

            hist1 = model.fit_generator(tr_gen_patch_mixup_warmup,
                                        steps_per_epoch=tr_gen_patch_mixup_warmup.nb_iterations,
                                        epochs=params_learn.get('mixup_warmup_epochs'),
                                        validation_data=val_gen_patch,
                                        validation_steps=val_gen_patch.nb_iterations,
                                        class_weight=class_weights_dict,
                                        workers=4,
                                        verbose=2,
                                        callbacks=callback_list)

            # watch warmup****************************************** we apply mixup now
            print('\n===implementing a warmup period for mixup: FINAL STAGE (with mixup)**************************')
            # delete the previous generator to free memory
            del tr_gen_patch_mixup_warmup

            params_learn['mixup'] = True
            tr_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                              file_list=tr_files,
                                              params_learn=params_learn,
                                              params_extract=params_extract,
                                              suffix_in='_mel',
                                              suffix_out='_label',
                                              floatx=np.float32
                                              )

            # callbacks
            if params_learn.get('early_stop') == "val_acc":
                early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
            elif params_learn.get('early_stop') == "val_loss":
                early_stop = EarlyStopping(monitor='val_loss', patience=params_learn.get('patience'), min_delta=0,
                                           verbose=1)
            checkpoint = ModelCheckpoint(params_files.get('save_model'), monitor='val_acc', verbose=1, save_best_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
            path_tb = 'logs/pics/tb/{}/v{}'.format(params_ctrl.get('output_file'), str(params_ctrl.get('count_trial')))
            tensorboard = TensorBoard(log_dir=path_tb,
                                      histogram_freq=0,
                                      batch_size=params_learn.get('batch_size'),
                                      write_grads=True,
                                      write_images=True,
                                      )
            callback_list = [checkpoint, early_stop, tensorboard, reduce_lr]

            hist2 = model.fit_generator(tr_gen_patch,
                                        steps_per_epoch=tr_gen_patch.nb_iterations,
                                        initial_epoch=params_learn.get('mixup_warmup_epochs'),
                                        epochs=params_learn.get('n_epochs'),
                                        validation_data=val_gen_patch,
                                        validation_steps=val_gen_patch.nb_iterations,
                                        class_weight=class_weights_dict,
                                        workers=4,
                                        verbose=2,
                                        callbacks=callback_list)

            hist1.history['acc'].extend(hist2.history['acc'])
            hist1.history['val_acc'].extend(hist2.history['val_acc'])
            hist1.history['loss'].extend(hist2.history['loss'])
            hist1.history['val_loss'].extend(hist2.history['val_loss'])
            hist1.history['lr'].extend(hist2.history['lr'])
            hist = hist1

        elif params_learn.get('stages') == 1:

            opt = Adam(lr=params_learn.get('lr'))
            model.compile(optimizer=opt, loss=params_loss.get('type'), metrics=['accuracy'])
            model.summary()

            # callbacks
            if params_learn.get('early_stop') == "val_acc":
                early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
            elif params_learn.get('early_stop') == "val_loss":
                early_stop = EarlyStopping(monitor='val_loss', patience=params_learn.get('patience'), min_delta=0,
                                           verbose=1)

            # vip save one best model for every fold, as I need this for submission
            params_files['save_model'] = os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                                      str(params_ctrl.get('count_trial')) + '_f' + str(kfo) + '.h5')
            checkpoint = ModelCheckpoint(params_files.get('save_model'), monitor='val_acc', verbose=1, save_best_only=True)

            reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
            # min_delta is not supported by LRplateau?
            path_tb = 'logs/pics/tb/{}/v{}'.format(params_ctrl.get('output_file'), str(params_ctrl.get('count_trial')))
            tensorboard = TensorBoard(log_dir=path_tb,
                                      histogram_freq=0,
                                      batch_size=params_learn.get('batch_size'),
                                      write_grads=True,
                                      write_images=True,
                                      )
            callback_list = [checkpoint, early_stop, tensorboard, reduce_lr]

            hist = model.fit_generator(tr_gen_patch,
                                       steps_per_epoch=tr_gen_patch.nb_iterations,
                                       epochs=params_learn.get('n_epochs'),
                                       validation_data=val_gen_patch,
                                       validation_steps=val_gen_patch.nb_iterations,
                                       class_weight=class_weights_dict,
                                       workers=4,
                                       verbose=2,
                                       callbacks=callback_list)
                                        # max_queue_size=20)

        # ------------------------------------------------

        # save learning curves as png plot, and a dict with the history values over epochs for diagnosis
        utils_classif.save_learning_curves(params_ctrl=params_ctrl, history=hist)
        utils_classif.save_history(params_ctrl=params_ctrl, history=hist.history)

    else:
        model = load_model(params_files.get('load_model'))
    # at this point, model is trained (or loaded)

    # ==================================================================================================== PREDICT
    # ==================================================================================================== PREDICT

    print('\nCompute predictions on test split, and save them in csv:==============================================\n')

    # to store prediction probabilites
    te_preds = np.empty((len(te_files), params_learn.get('n_classes')))

    # init: create tensor with all T_F patches from test set for prediction
    # grab every T_F rep file (computed on the file level)
    # split it in T_F patches and store it in tensor, sorted by file
    # TODO update prototype with last params
    te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_dev'),
                                         file_list=te_files,
                                         params_extract=params_extract,
                                         suffix_in='_mel',
                                         floatx=np.float32,
                                         scaler=tr_gen_patch.scaler
                                         )

    # softmax_test_set = [None] * len(te_files)
    for i in trange(len(te_files), miniters=int(len(te_files) / 100), ascii=True, desc="Predicting..."):
        # return all patches for a sound file, every time get_patches_file() is called, file index is increased
        patches_file = te_gen_patch.get_patches_file()
        # ndarray with (nb_patches_per_file, 1, time, freq).
        # this can be thought of as a batch (for evaluation), with all the patches from a sound file
        # this could be a fixed number of patches in the fix mode, or a variable number in the varXX modes

        # predicting now on the T_F patch level (not on the wav clip-level)
        preds_patch_list = model.predict(patches_file).tolist()
        # softmax values, similar to probabilities, in a list of floats per patch
        preds_patch = np.array(preds_patch_list)

        # save softmax values: a list of len(te_files) ndarrays
        # softmax_test_set[i] = preds_patch

        # aggregate softmax values across patches in order to produce predictions on the file/clip level
        if params_learn.get('predict_agg') == 'amean':
            preds_file = np.mean(preds_patch, axis=0)
        elif params_recog.get('aggregate') == 'gmean':
            preds_file = gmean(preds_patch, axis=0)
        else:
            print('unkown aggregation method for prediction')
        te_preds[i, :] = preds_file

    # todo: not now. utils_classif.save_softmax(params_ctrl=params_ctrl, values=softmax_test_set)

    list_labels = np.array(list_labels)
    # grab the label with highest score
    # vip this is what I need to report [0: 10]
    pred_label_files_int = np.argmax(te_preds, axis=1)
    pred_labels = [int_to_label[x] for x in pred_label_files_int]


    # create dataframe with predictions
    # columns: fname & label
    # this is based on the features file, instead on the wav file (extraction errors could occur)
    te_files_wav = [f.replace(suffix_in + '.data', '.wav') for f in te_files]
    if not os.path.isfile(params_files.get('predictions')):
        # fold 1: create the predictions file
        pred = pd.DataFrame(te_files_wav, columns=['fname'])
        pred['label'] = pred_labels
        pred['label_int'] = pred_label_files_int
        pred.to_csv(params_files.get('predictions'), index=False)
        del pred

    else:
        # fold > 1. There is already a predictions file
        pred = pd.read_csv(params_files.get('predictions'))
        old_fname = pred.fname.values.tolist()
        old_label = pred.label.values.tolist()
        old_label_int = pred.label_int.values.tolist()

        new_pred_fname = old_fname + te_files_wav
        new_pred_label = old_label + pred_labels
        new_pred_label_int = old_label_int + pred_label_files_int.tolist()

        del pred
        pred = pd.DataFrame(new_pred_fname, columns=['fname'])
        pred['label'] = new_pred_label
        pred['label_int'] = new_pred_label_int
        pred.to_csv(params_files.get('predictions'), index=False)

    # deleter variables from past fold to free memory.
    del tr_gen_patch
    del val_gen_patch
    # this model was trained on split X, and no need anymore
    del model

# vip once we are done with all the 4 folds
# # =================================================================================================== EVAL
# # =================================================================================================== EVAL
print('\nEvaluate ACC and print score for the cross validation setup============================================\n')

# read ground truth: gt_dev

# init Evaluator object
evaluator = Evaluator(gt_dev, pred, list_labels, params_ctrl, params_files)

print('\n=============================ACCURACY===============================================================')
print('=============================ACCURACY===============================================================\n')
evaluator.evaluate_acc()
evaluator.evaluate_acc_classwise()

# evaluator.print_summary_eval()

end = time.time()
print('\n=============================Job finalized, but lacks DCASE metrics========================================\n')
print('\nTime elapsed for the job: %7.2f hours' % ((end - start) / 3600.0))
print('\n====================================================================================================\n')


print('\n====================Starting metrics for challenge with REAL frontend=====================================')
print('====================Starting metrics for challenge with REAL frontend=====================================')
print('====================Starting metrics for challenge with REAL frontend=====================================\n')

data_folder_path = '../data/foa_dev/'
# Iterate over all audio files from the dev set, some are from split 1234
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Path stuff

# This parameter will define the algorithm type
preset_string = 'Q'

# Default preset: contains path to folders
params = get_params(preset_string)

# Dataset type:
dataset_type_folder = params['dataset'] + "_" + params['mode']
dataset_preset_folder = dataset_type_folder + '_' + preset_string

# Get folder names before and after classification
doa_folder = params['before_classification_folder_name']
classif_folder = params['after_classification_folder_name']

# Path to audio folder
dataset_dir = '../data'
data_folder_path = os.path.join(dataset_dir, dataset_type_folder)

# Path to results_metadata folder _before classification_; it should exist
results_metadata_doa_folder = os.path.join('.' + params['metadata_result_folder_path'],
                                           dataset_preset_folder,
                                           doa_folder)
if not os.path.exists(results_metadata_doa_folder):
    os.mkdir(results_metadata_doa_folder)

# Path to results_metadata folder _before classification_; create it if necessary
results_metadata_classif_folder = os.path.join('.' + params['metadata_result_folder_path'],
                                               dataset_preset_folder,
                                               classif_folder)
if not os.path.exists(results_metadata_classif_folder):
    os.mkdir(results_metadata_classif_folder)

# Path to results_output folder _before classification_; it should exist
results_output_doa_folder = os.path.join('.' + params['output_result_folder_path'],
                                           dataset_preset_folder,
                                           doa_folder)
if not os.path.exists(results_output_doa_folder):
    os.mkdir(results_output_doa_folder)

# Path to results_output folder _before classification_; create it if necessary
results_output_classif_folder = os.path.join('.' + params['output_result_folder_path'],
                                               dataset_preset_folder,
                                               classif_folder)
if not os.path.exists(results_output_classif_folder):
    os.mkdir(results_output_classif_folder)

# vip load best model for every fold, as I need this for submission
model_f1 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f1.h5'))
model_f2 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f2.h5'))
model_f3 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f3.h5'))
model_f4 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f4.h5'))

sr = 48000
for audio_file_name in audio_files:
    # always all the clips from the entire dataset (this is fixed)

    # Get associated metadata file
    metadata_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']
    # this csv contains the list of segmented events from the parent audio clip, in this case, segmented with the
    # REAL info provided by andres frontend
    # hence REAL conditions. vip WE need to use the path given by results_metadata_doa_folder defined above
    # to get the correct file

    # This is our modified metadata result array
    metadata_result_classif_array = []

    # Iterate over the associated doa metadata file
    with open(os.path.join(results_metadata_doa_folder, metadata_file_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Discard the first line (just the column titles)
            if i > 0:
                # Get values for this sound event
                sound_class_string = row[0]
                start_time_seconds = float(row[1])
                end_time_seconds = float(row[2])

                # Slice the b_format audio to the corresponding event length
                start_frame = int(np.floor(start_time_seconds * sr))
                end_frame = int(np.ceil(end_time_seconds * sr))

                # from one event entry in the csv, to its corresponding audio clip filename (that I stored previously)
                # filename = sound_class_string + '_' + str(start_frame) + '_' + metadata_file_name.split('.')[0] + '.wav' old simpler name
                # new name
                filename = sound_class_string + '_' + str(start_frame) + '_' + str(end_frame) + '_' + metadata_file_name.split('.')[0] + '.wav'

                curent_split = int(filename.split('_')[3][-1])

                # vip Classify: this will need 4 models for 4 test splits in x-val in development mode + one model for evaluation mode
                # to store prediction probabilites for one single test clip
                te_preds = np.empty((1, params_learn.get('n_classes')))

                # only the file under question currently
                ff_list_dev_param = [filename.replace('.wav', suffix_in + '.data')]

                # grab scaler from the four I saved in x-val
                current_scaler = scalers[curent_split - 1]

                # vip: we are in REAL conditions, hence the dataset is 'dev_param' (and not dev). Watch path
                te_param_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_dev_param'),
                                                           file_list=ff_list_dev_param,
                                                           params_extract=params_extract,
                                                           suffix_in='_mel',
                                                           floatx=np.float32,
                                                           scaler=current_scaler
                                                           )

                # return all patches for a sound file, every time get_patches_file() is called, file index is increased
                patches_file = te_param_gen_patch.get_patches_file()
                # ndarray with (nb_patches_per_file, 1, time, freq).
                # this can be thought of as a batch (for evaluation), with all the patches from a sound file

                # choose model accordingly
                # predicting now on the T_F patch level (not on the wav clip-level)
                if curent_split == 1:
                    preds_patch_list = model_f1.predict(patches_file).tolist()
                elif curent_split == 2:
                    preds_patch_list = model_f2.predict(patches_file).tolist()
                elif curent_split == 3:
                    preds_patch_list = model_f3.predict(patches_file).tolist()
                elif curent_split == 4:
                    preds_patch_list = model_f4.predict(patches_file).tolist()

                # softmax values, similar to probabilities, in a list of floats per patch
                preds_patch = np.array(preds_patch_list)
#
                # aggregate softmax values across patches in order to produce predictions on the file/clip level
                if params_learn.get('predict_agg') == 'amean':
                    preds_file = np.mean(preds_patch, axis=0)
                elif params_recog.get('aggregate') == 'gmean':
                    preds_file = gmean(preds_patch, axis=0)
                else:
                    print('unkown aggregation method for prediction')
                te_preds[0, :] = preds_file

                # vip this is what I need to report [0: 10]
                class_id = np.argmax(te_preds, axis=1)
                # Substitute the None for the current class, and append to the new metadata array
                row[0] = class_id
                metadata_result_classif_array.append(row)
#
    # Write a new results_metadata_classif file with the modified classes
    metadata_result_classif_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']
    path_to_write = os.path.join(results_metadata_classif_folder, metadata_result_classif_file_name)
    write_metadata_result_file(metadata_result_classif_array, path_to_write)

    # Write a new result_output_classif file with the modified classes
    output_result_classif_dict = build_result_dict_from_metadata_array(metadata_result_classif_array, params['required_window_hop'])
    path_to_write = os.path.join(results_output_classif_folder, metadata_file_name)
    write_output_result_file(output_result_classif_dict, path_to_write)


print('-------------- COMPUTE DOA METRICS REAL--------------')
gt_folder = os.path.join(dataset_dir, 'metadata_'+params['mode'])
compute_DOA_metrics(gt_folder, results_output_classif_folder)
#
#
#
print('\n====================Starting metrics for challenge with IDEAL frontend=====================================')
print('====================Starting metrics for challenge with IDEAL frontend=====================================')
print('====================Starting metrics for challenge with IDEAL frontend=====================================')
print('====================Starting metrics for challenge with IDEAL frontend=====================================\n')


# Path to results_metadata folder _before classification_; it should exist
results_metadata_doa_folder = os.path.join('.' + params['metadata_result_folder_path'],
                                           'metadata_dev',
                                           doa_folder)
if not os.path.exists(results_metadata_doa_folder):
    os.mkdir(results_metadata_doa_folder)

# Path to results_metadata folder _before classification_; create it if necessary
results_metadata_classif_folder = os.path.join('.' + params['metadata_result_folder_path'],
                                               'metadata_dev',
                                               classif_folder)
if not os.path.exists(results_metadata_classif_folder):
    os.mkdir(results_metadata_classif_folder)

# Path to results_output folder _before classification_; it should exist
results_output_doa_folder = os.path.join('.' + params['output_result_folder_path'],
                                         'metadata_dev',
                                         doa_folder)
if not os.path.exists(results_output_doa_folder):
    os.mkdir(results_output_doa_folder)

# Path to results_output folder _before classification_; create it if necessary
results_output_classif_folder = os.path.join('.' + params['output_result_folder_path'],
                                             'metadata_dev',
                                             classif_folder)
if not os.path.exists(results_output_classif_folder):
    os.mkdir(results_output_classif_folder)

# vip load best model for every fold, as I need this for submission
model_f1 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f1.h5'))
model_f2 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f2.h5'))
model_f3 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f3.h5'))
model_f4 = load_model(os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                   str(params_ctrl.get('count_trial')) + '_f4.h5'))

sr = 48000
for audio_file_name in audio_files:
    # always all the clips from the entire dataset (this is fixed)

    # Get associated metadata file
    metadata_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']
    # this csv contains the list of segmented events from the parent audio clip, in this case, segmented with the GT
    # info, hence ideal conditions. vip WE need to use the path given by results_metadata_doa_folder defined above
    # to get the correct file

    # This is our modified metadata result array
    metadata_result_classif_array = []

    # Iterate over the associated doa metadata file
    with open(os.path.join(results_metadata_doa_folder, metadata_file_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Discard the first line (just the column titles)
            if i > 0:
                # Get values for this sound event
                sound_class_string = row[0]
                start_time_seconds = float(row[1])
                end_time_seconds = float(row[2])

                # Slice the b_format audio to the corresponding event length
                start_frame = int(np.floor(start_time_seconds * sr))
                end_frame = int(np.ceil(end_time_seconds * sr))

                # from one event entry in the csv, to its corresponding audio clip filename (that I stored previously)
                filename = sound_class_string + '_' + str(start_frame) + '_' + metadata_file_name.split('.')[0] + '.wav'
                curent_split = int(filename.split('_')[2][-1])

                # vip Classify: this will need 4 models for 4 test splits in x-val in development mode + one model for evaluation mode
                # to store prediction probabilites for one single test clip
                te_preds = np.empty((1, params_learn.get('n_classes')))

                # only the file under question currently
                ff_list_dev_ideal = [filename.replace('.wav', suffix_in + '.data')]

                # grab scaler from the four I saved in x-val
                current_scaler = scalers[curent_split - 1]

                # vip: we are in ideal conditions, hence the dataset is 'dev' (and not dev_param). Watch path
                te_idal_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_dev'),
                                                           file_list=ff_list_dev_ideal,
                                                           params_extract=params_extract,
                                                           suffix_in='_mel',
                                                           floatx=np.float32,
                                                           scaler=current_scaler
                                                           )

                # return all patches for a sound file, every time get_patches_file() is called, file index is increased
                patches_file = te_idal_gen_patch.get_patches_file()
                # ndarray with (nb_patches_per_file, 1, time, freq).
                # this can be thought of as a batch (for evaluation), with all the patches from a sound file

                # choose model accordingly
                # predicting now on the T_F patch level (not on the wav clip-level)
                if curent_split == 1:
                    preds_patch_list = model_f1.predict(patches_file).tolist()
                elif curent_split == 2:
                    preds_patch_list = model_f2.predict(patches_file).tolist()
                elif curent_split == 3:
                    preds_patch_list = model_f3.predict(patches_file).tolist()
                elif curent_split == 4:
                    preds_patch_list = model_f4.predict(patches_file).tolist()

                # softmax values, similar to probabilities, in a list of floats per patch
                preds_patch = np.array(preds_patch_list)

                # aggregate softmax values across patches in order to produce predictions on the file/clip level
                if params_learn.get('predict_agg') == 'amean':
                    preds_file = np.mean(preds_patch, axis=0)
                elif params_recog.get('aggregate') == 'gmean':
                    preds_file = gmean(preds_patch, axis=0)
                else:
                    print('unkown aggregation method for prediction')
                te_preds[0, :] = preds_file

                # vip this is what I need to report [0: 10]
                class_id = np.argmax(te_preds, axis=1)
                # Substitute the None for the current class, and append to the new metadata array
                row[0] = class_id
                metadata_result_classif_array.append(row)

    # Write a new results_metadata_classif file with the modified classes
    metadata_result_classif_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']
    path_to_write = os.path.join(results_metadata_classif_folder, metadata_result_classif_file_name)
    write_metadata_result_file(metadata_result_classif_array, path_to_write)

    # Write a new result_output_classif file with the modified classes
    output_result_classif_dict = build_result_dict_from_metadata_array(metadata_result_classif_array, params['required_window_hop'])
    path_to_write = os.path.join(results_output_classif_folder, metadata_file_name)
    write_output_result_file(output_result_classif_dict, path_to_write)


print('-------------- COMPUTE DOA METRICS IDEAL--------------')
gt_folder = os.path.join(dataset_dir, 'metadata_'+params['mode'])
compute_DOA_metrics(gt_folder, results_output_classif_folder)

print('\n=============================Job finalized for real==========================================================\n')
print('====================================================================================================')
print('====================================================================================================')
