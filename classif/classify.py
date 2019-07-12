
import os

# basic
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tqdm import tqdm, trange
import time
import pprint
import datetime
import argparse
from scipy.stats import gmean
import yaml
import shutil

# keras
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# DIY
import utils_classif
from feat_ext import load_audio_file, get_mel_spectrogram, modify_file_variable_length
from data import get_label_files, DataGeneratorPatch, PatchGeneratorPerFile
from architectures import get_model_crnn_seld_tagger
from eval import Evaluator

import csv
import sys
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
parser = argparse.ArgumentParser(description='DCASE2019 Task3')
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

# determine loss function for stage 1 (or entire training)
if params_loss.get('type') == 'CCE':
    params_loss['type'] = 'categorical_crossentropy'
elif params_loss.get('type') == 'MAE':
    params_loss['type'] = 'mean_absolute_error'

params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))

#  vip to deploy. for public, put directly params_ctrl.gt('dataset_path') within params_path
path_root_data = params_ctrl.get('dataset_path')

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
                'gt_dev': os.path.join(params_path.get('gt_files'), 'gt_dev.csv')}

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
filelist_audio_dev = gt_dev.fname.values.tolist()

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


        # do feature extraction for dev_param (outcome of complete parametric frontend)========================================
        # do feature extraction for dev_param (outcome of complete parametric frontend)========================================
        audio_files_dev_param = [f for f in os.listdir(params_path.get('audiopath_dev_param')) if not f.startswith('.')]

        nb_files_dev_param = len(audio_files_dev_param)
        if not os.path.exists(params_path.get('featurepath_dev_param')) or \
                        len(os.listdir(params_path.get('featurepath_dev_param'))) < nb_files_dev_param * 0.8:

            if os.path.exists(params_path.get('featurepath_dev_param')):
                shutil.rmtree(params_path.get('featurepath_dev_param'))
            os.makedirs(params_path.get('featurepath_dev_param'))

            print(
                '\nFeature extraction for dev set parametric (outcome of parametric frontend). Features dumped in {}.........................'.
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
                    mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                    # save the T_F rep to a binary file (only the considered length)
                    utils_classif.save_tensor(var=mel_spectrogram,
                                              out_path=os.path.join(params_path.get('featurepath_dev_param'),
                                                                    f_name.replace('.wav', '.data')), suffix='_mel')

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


# select the subset of training data to consider: all, clean, noisy, noisy_small
# =====================================================================================================================
# =====================================================================================================================

ff_list_dev = [filelist_audio_dev[i].replace('.wav', suffix_in + '.data') for i in range(len(filelist_audio_dev))]
labels_audio_dev = get_label_files(filelist=ff_list_dev,
                                   dire=params_path.get('featurepath_dev'),
                                   suffix_in=suffix_in,
                                   suffix_out=suffix_out
                                   )

print('Number of clips considered as dev set: {0}'.format(len(ff_list_dev)))
print('Number of labels loaded for dev set: {0}'.format(len(labels_audio_dev)))

scalers = [None]*4
# determine the validation setup according to the folds, and perform training / val / test for each fold
for kfo in range(1, 5):
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

    # ============================================================DEFINE AND FIT A MODEL
    # ============================================================DEFINE AND FIT A MODEL

    tr_loss, val_loss = [0] * params_learn.get('n_epochs'), [0] * params_learn.get('n_epochs')
    # ============================================================
    if params_ctrl.get('learn'):
        if params_learn.get('model') == 'crnn_seld_tagger':
            model = get_model_crnn_seld_tagger(params_crnn=params_crnn, params_learn=params_learn,
                                               params_extract=params_extract)

        if params_learn.get('stages') == 1:

            opt = Adam(lr=params_learn.get('lr'))
            model.compile(optimizer=opt, loss=params_loss.get('type'), metrics=['accuracy'])
            model.summary()

            # callbacks
            if params_learn.get('early_stop') == "val_acc":
                early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
            elif params_learn.get('early_stop') == "val_loss":
                early_stop = EarlyStopping(monitor='val_loss', patience=params_learn.get('patience'), min_delta=0,
                                           verbose=1)

            # save one best model for every fold, as needed for submission
            params_files['save_model'] = os.path.join(path_trained_models, params_ctrl.get('output_file') + '_v' +
                                                      str(params_ctrl.get('count_trial')) + '_f' + str(kfo) + '.h5')
            checkpoint = ModelCheckpoint(params_files.get('save_model'), monitor='val_acc', verbose=1, save_best_only=True)

            reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
            callback_list = [checkpoint, early_stop, reduce_lr]

            hist = model.fit_generator(tr_gen_patch,
                                       steps_per_epoch=tr_gen_patch.nb_iterations,
                                       epochs=params_learn.get('n_epochs'),
                                       validation_data=val_gen_patch,
                                       validation_steps=val_gen_patch.nb_iterations,
                                       class_weight=class_weights_dict,
                                       workers=4,
                                       verbose=2,
                                       callbacks=callback_list)

    # ==================================================================================================== PREDICT
    # ==================================================================================================== PREDICT

    print('\nCompute predictions on test split, and save them in csv:==============================================\n')

    # to store prediction probabilites
    te_preds = np.empty((len(te_files), params_learn.get('n_classes')))

    te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_dev'),
                                         file_list=te_files,
                                         params_extract=params_extract,
                                         suffix_in='_mel',
                                         floatx=np.float32,
                                         scaler=tr_gen_patch.scaler
                                         )

    for i in trange(len(te_files), miniters=int(len(te_files) / 100), ascii=True, desc="Predicting..."):
        patches_file = te_gen_patch.get_patches_file()

        preds_patch_list = model.predict(patches_file).tolist()
        preds_patch = np.array(preds_patch_list)

        if params_learn.get('predict_agg') == 'amean':
            preds_file = np.mean(preds_patch, axis=0)
        elif params_recog.get('aggregate') == 'gmean':
            preds_file = gmean(preds_patch, axis=0)
        else:
            print('unkown aggregation method for prediction')
        te_preds[i, :] = preds_file

    list_labels = np.array(list_labels)
    pred_label_files_int = np.argmax(te_preds, axis=1)
    pred_labels = [int_to_label[x] for x in pred_label_files_int]

    te_files_wav = [f.replace(suffix_in + '.data', '.wav') for f in te_files]
    if not os.path.isfile(params_files.get('predictions')):
        # fold 1: create the predictions file
        pred = pd.DataFrame(te_files_wav, columns=['fname'])
        pred['label'] = pred_labels
        pred['label_int'] = pred_label_files_int
        pred.to_csv(params_files.get('predictions'), index=False)
        del pred

    else:
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

# init Evaluator object
evaluator = Evaluator(gt_dev, pred, list_labels, params_ctrl, params_files)

print('\n=============================ACCURACY===============================================================')
print('=============================ACCURACY===============================================================\n')
evaluator.evaluate_acc()
evaluator.evaluate_acc_classwise()

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
# old: this overwrites the several trials
results_output_classif_folder = os.path.join('.' + params['output_result_folder_path'],
                                               dataset_preset_folder,
                                               classif_folder)
# create just the folder classif if there is not such thing
if not os.path.exists(results_output_classif_folder):
    os.mkdir(results_output_classif_folder)

# new: a folder for each trial. This is already what we have to submit for development mode
results_output_classif_folder = os.path.join('.' + params['output_result_folder_path'],
                                               dataset_preset_folder,
                                               classif_folder,
                                             params_ctrl.get('output_file') + '_v' + str(params_ctrl.get('count_trial')))
if not os.path.exists(results_output_classif_folder):
    os.mkdir(results_output_classif_folder)

# load best model for every fold, for submission
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

    # Get associated metadata file
    metadata_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']

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
                filename = sound_class_string + '_' + str(start_frame) + '_' + str(end_frame) + '_' + metadata_file_name.split('.')[0] + '.wav'
                curent_split = int(filename.split('_')[3][-1])

                # Classify: this will need 4 models for 4 test splits in x-val in development mode + one model for evaluation mode
                te_preds = np.empty((1, params_learn.get('n_classes')))

                # only the file under question
                ff_list_dev_param = [filename.replace('.wav', suffix_in + '.data')]
                current_scaler = scalers[curent_split - 1]

                te_param_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_dev_param'),
                                                           file_list=ff_list_dev_param,
                                                           params_extract=params_extract,
                                                           suffix_in='_mel',
                                                           floatx=np.float32,
                                                           scaler=current_scaler
                                                           )

                patches_file = te_param_gen_patch.get_patches_file()

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

                preds_patch = np.array(preds_patch_list)

                # aggregate softmax values across patches in order to produce predictions on the file/clip level
                if params_learn.get('predict_agg') == 'amean':
                    preds_file = np.mean(preds_patch, axis=0)
                elif params_recog.get('aggregate') == 'gmean':
                    preds_file = gmean(preds_patch, axis=0)
                else:
                    print('unkown aggregation method for prediction')
                te_preds[0, :] = preds_file

                class_id = np.argmax(te_preds, axis=1)
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

# load best model for every fold, for submission
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

    # Get associated metadata file
    metadata_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']
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

                # Classify: this will need 4 models for 4 test splits in x-val in development mode + one model for evaluation mode
                # to store prediction probabilites for one single test clip
                te_preds = np.empty((1, params_learn.get('n_classes')))

                # only the file under question
                ff_list_dev_ideal = [filename.replace('.wav', suffix_in + '.data')]
                current_scaler = scalers[curent_split - 1]
                te_idal_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_dev'),
                                                           file_list=ff_list_dev_ideal,
                                                           params_extract=params_extract,
                                                           suffix_in='_mel',
                                                           floatx=np.float32,
                                                           scaler=current_scaler
                                                           )

                patches_file = te_idal_gen_patch.get_patches_file()

                if curent_split == 1:
                    preds_patch_list = model_f1.predict(patches_file).tolist()
                elif curent_split == 2:
                    preds_patch_list = model_f2.predict(patches_file).tolist()
                elif curent_split == 3:
                    preds_patch_list = model_f3.predict(patches_file).tolist()
                elif curent_split == 4:
                    preds_patch_list = model_f4.predict(patches_file).tolist()

                preds_patch = np.array(preds_patch_list)

                # aggregate softmax values across patches in order to produce predictions on the file/clip level
                if params_learn.get('predict_agg') == 'amean':
                    preds_file = np.mean(preds_patch, axis=0)
                elif params_recog.get('aggregate') == 'gmean':
                    preds_file = gmean(preds_patch, axis=0)
                else:
                    print('unkown aggregation method for prediction')
                te_preds[0, :] = preds_file

                class_id = np.argmax(te_preds, axis=1)
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

print('\n=============================Job finalized==========================================================\n')
print('====================================================================================================')
print('====================================================================================================')
