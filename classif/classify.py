
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
    get_mobilenet_ka19
from eval import Evaluator
from losses import crossentropy_cochlear, crossentropy_diy_max, lq_loss, lq_loss_wrap, crossentropy_diy_max_wrap, \
    crossentropy_diy_outlier_wrap, crossentropy_reed_wrap, crossentropy_diy_outlier_origin_wrap
from mobilenet import mobilenet


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
               'featuredir_dev': 'audio_dev_varup1/',
               'featuredir_eval': 'audio_eval_varup1/',
               'path_to_dataset': path_root_data,
               'audiodir_dev': 'wav/dev/',
               'audiodir_eval': 'wav/eval/',
               'audio_shapedir_dev': 'audio_dev_shapes/',
               'audio_shapedir_eval': 'audio_eval_shapes/',
               'gt_files': path_root_data}

if params_learn.get('mixup_log'):
    # we will be computing log of spectrograms on the fly for training (after mixup)
    # for the test set, the log is applied pre-computed
    # here we define a new folder for the train features, where NO log is applied
    params_path['featuredir_dev'] = 'audio_dev_varup2_nolog/'
    print('\n===Using dev set WITHOUT logarithm on the spectrograms.')
    # since this folder is non existing, it will compute features, with the log disabled in the feat extraction

params_path['featurepath_dev'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_dev'))
params_path['featurepath_eval'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_eval'))

params_path['audiopath_dev'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_dev'))
params_path['audiopath_eval'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_eval'))

params_path['audio_shapedir_dev'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_dev'))
params_path['audio_shapedir_eval'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_eval'))


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
#
# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# compute T_F representation
# mel-spectrogram for all files in the dataset and store it
var_lens = {item: [] for item in label_to_int.keys()}
var_lens['overall'] = []
if params_ctrl.get('feat_ext'):
    if params_ctrl.get('pipeline') == 'T_F':
        n_extracted_dev = 0; n_extracted_te = 0; n_failed_dev = 0; n_failed_te = 0

        # only if features have not been extracted, ie
        # if folder does not exist, or it exists with less than 80% of the feature files
        # create folder and extract features
        nb_files_dev = len(filelist_audio_dev)
        if not os.path.exists(params_path.get('featurepath_dev')) or \
                        len(os.listdir(params_path.get('featurepath_dev'))) < nb_files_dev*0.8:

            if os.path.exists(params_path.get('featurepath_dev')):
                shutil.rmtree(params_path.get('featurepath_dev'))
            os.makedirs(params_path.get('featurepath_dev'))

            print('\nFeature extraction for dev set (prints enabled)..........................................')
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

# vip determine the validation setup according to the folds, and perform training / val / test for each fold
# for kfo in range(1, 5):
for kfo in range(1, 2):
    print('\n===Processing fold {} within the x-val setup...'.format(kfo))
    print('=========================================================================================\n')

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

    tr_files0 = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_tr[0]]
    tr_files1 = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_tr[1]]
    tr_files = tr_files0 + tr_files1
    val_files = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_val[0]]
    te_files = [fname for idx, fname in enumerate(ff_list_dev) if splitlist_audio_dev[idx] == splits_te[0]]

    # SC
    if len(tr_files) + len(val_files) + len(te_files) != len(ff_list_dev):
        print('You messed up in x-val setup for fold: {0}'.format(len(kfo)))
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

    # deleter variables from past fold to free memory
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
print('\n=============================Job finalized==========================================================\n')
print('\nTime elapsed for the job: %7.2f hours' % ((end - start) / 3600.0))
print('\n====================================================================================================\n')
