
import numpy as np
import os
import utils_classif
from numpy.random import permutation
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence, to_categorical
import librosa
import matplotlib.pyplot as plt

from feat_ext import load_audio_file, get_normalized_audio, modify_file_variable_length

"""
TODO:
data generators are thought for small datasets (no memory issues). improve by optimizing resources for large datasets
"""

def get_prior_distribution_noise(gt_label, num_classes, idx_noisy_classes, epsi, delta_k):
    """
    NOTE the processing can be done:
    -watch, we process datapoint by datapoint, ie one row at a time, with a new epsi for a patch label
    -or with all the batch at once

    # gt_label can be the entire matrix (batch_size, n_classes) (in mode all)
    # or only one patch (1, n_classes) in the origin mode (every patch comes from the noisy or clean set), or if there is delta_epsi

    :param gt_label:
    :param num_classes:
    :param idx_noisy_classes:
    :param epsi:
    :param delta_k:
    :return: vector with the distribution for the ground truth vector
    """

    # vip 2) define distribution of (1-epsilon)
    # fix create function: code repeated, pero aqui toy lidiando con un vector
    # fix create function: code repeated, pero aqui toy lidiando con una matrix 64,20
    distri_prior_noise = np.zeros_like(gt_label)
    if distri_prior_noise.shape[0] == num_classes:
        distri_prior_noise.shape = (1, 20)

    # 20 columns ie classes
    # high noise, less trustable, less value
    # dejalo montado para que valga para matrix y vector fix
    distri_prior_noise[:, idx_noisy_classes] = epsi / (num_classes + delta_k)

    # low noise, more trustable, more value
    complement_clean_classes = np.array([val for val in range(num_classes) if val not in idx_noisy_classes])
    # dejalo montado para que valga para matrix y vector fix
    distri_prior_noise[:, complement_clean_classes] = epsi / (num_classes - delta_k)

    # but chances are sum(prior_noise) is not epsilon, so we normalize
    distri_prior_noise = distri_prior_noise * epsi / np.sum(distri_prior_noise[0, :])
    if np.sum(distri_prior_noise[0, :]) <= epsi - 0.0001 or np.sum(distri_prior_noise[0, :]) > epsi + 0.0001:
        print("Warning: distri_prior_noise is not calculated well, does not add up to epsilon ")

    return distri_prior_noise


def get_prior_distribution_patch(patch_prior, epsi):
    """

    :param patch_prior: number of pathces per class in the train set (without val set)
    :param epsi:
    :return:
    """
    # this is a vector of integers. you have epsilon to spread considering those numbers, and they have to add epsilon

    # sum(patch_prior) is very large, we normalize to epsilon
    distri_prior_unigram = patch_prior * epsi / np.sum(patch_prior[0, :])
    if np.sum(distri_prior_unigram[0, :]) <= epsi - 0.0001 or np.sum(
            distri_prior_unigram[0, :]) > epsi + 0.0001:
        print("Warning: distri_prior_unigram is not calculated well, does not add up to epsilon ")
    return distri_prior_unigram


def label_smoothing(y_train_cat, eps_LSR_noisy=0, epsilon_clean=0, delta_eps_LSR=None, num_classes=0,
                    mode=None, distri_prior=None, delta_k=0, patch_prior=None, LSRmode=None, LSRmapping=None):
    """
    Classical label smoothing for the labels that have not been manually verified. Also applied to the labels that have
    been manually verified as not all annotators agreed on some sounds (use a much smaller value).
    :param y_train_cat: Categorical labels of the training data
    :param epsilon_weak: parameter of label smoothing for files that have not been manually verified
    :param epsilon_verified: parameter of label smoothing for files that have been manually verified
    :param num_classes: number of classes in the training set (get rid off? can be computed here....)
    delta_eps_LSR: delta epsilon to add to / substract from epsilon, based on a prior. This defines the final epsilon
    to be applied to the active class label

    distri_prior:
    None = uniform; uniform distribution: prior distribution over labels = 1/K
    noise = distribution at two levels of noise, depending on prior level of noise in each category
    patch: distribution at "infinite" levels, based on the frequency of occurrence of patchs (not clips) in every class


    delta_k: is a delta with respecto to the number of classes for the operation epsilon/(K +- delta_k) when using
    distribution based on noise prior. Tpiycal values by inspection, considering K=20: 2,4,6,8. 6 seems nice.
    :return:
    """

    # vip y_train_cat can be the entire matrix (batch_size, n_classes) (in mode all)
    # vip or only one patch (1, n_classes) in the origin mode (every patch comes from the noisy or clean set)
    if mode == 'all' or mode == 'noisy':

        # {'Acoustic_guitar': 0, 'Bass_guitar': 1, 'Clapping': 2, 'Coin_(dropping)': 3, 'Crash_cymbal': 4,
        # 'Dishes_and_pots_and_pans': 5, 'Engine': 6, 'Fart': 7, 'Fire': 8, 'Fireworks': 9, 'Glass': 10, 'Hi-hat': 11,
        # 'Piano': 12, 'Rain': 13, 'Slam': 14, 'Squeak': 15, 'Tearing': 16, 'Walk_or_footsteps': 17, 'Wind': 18, 'Writing': 19}

        # 1.1) GROUPS2: split classes in two groups based on noise, based on acoustic inspection in ICASSP19
        # bass guitar, clapping, crash, Engine, Fire, Rain, Slam, Walk, Wind
        idx_noisy_classes = [1, 2, 4, 6, 8, 13, 14, 17, 18]
        # create a vector with the noise priors, with the size of y_train_cat

        # improved group of noisy classes, to try
        # idx_noisy_classes_alpha = [1, 2, 4, 6, 8, 13, 14, 17, 18]

        # 1.2) GROUPS3: split classes in 3 groups based on noise, based on acoustic inspection in ICASSP19
        idx_lownoise_classes = [0, 5, 7, 10, 11, 12, 15, 16, 19]
        idx_midnoise_classes = [1, 2, 3, 4, 9]
        idx_highnoise_classes = [6, 8, 13, 14, 17, 18]

        if not distri_prior:
            # vip distri_prior = None: uniform distrubution: prior distribution over labels = 1/K
            if not delta_eps_LSR:
                # watch delta_eps_LSR is None, hence all classes have the same epsilon for the ACTIVE class label
                # Inception implementation: max is not 1-epsi, but 1- epsi + epsi/n_classes
                y_train_cat = y_train_cat*(1-eps_LSR_noisy) + eps_LSR_noisy/num_classes
                # we are doing a re-distribution of the energy, the label vector still adds up to 1,
                # hence further processing in the losses wr2 origin holds

            else:
                # watch delta_eps_LSR is a float, some classes have higher/lower epsilon for the ACTIVE class label
                if LSRmode == 'GROUPS2':
                    # print('\n==distri_prior = False; GROUPS2')
                    # 1.1) GROUPS2: split classes in two groups based on (more/less) noise. 2 levels of noisiness
                    for kk in range(y_train_cat.shape[0]):
                        if np.nonzero(y_train_cat[kk])[0] in idx_noisy_classes:
                            # very noisy, increase epsi
                            new_epsi = eps_LSR_noisy + delta_eps_LSR
                        else:
                            # low noise, decrease epsi
                            new_epsi = eps_LSR_noisy - delta_eps_LSR
                        y_train_cat[kk] = y_train_cat[kk]*(1-new_epsi) + new_epsi/num_classes

                elif LSRmode == 'GROUPS3':
                    # print('\n==distri_prior = False; GROUPS3')
                    # 1.2) GROUPS3: split classes into 3 groups based on noise. 3 levels of noisiness
                    for kk in range(y_train_cat.shape[0]):
                        if np.nonzero(y_train_cat[kk])[0] in idx_highnoise_classes:
                            # very noisy, increase epsi
                            new_epsi = eps_LSR_noisy + delta_eps_LSR
                        elif np.nonzero(y_train_cat[kk])[0] in idx_midnoise_classes:
                            # mid noisy, do nothing
                            new_epsi = eps_LSR_noisy
                        if np.nonzero(y_train_cat[kk])[0] in idx_lownoise_classes:
                            # low noise, decrease epsi
                            new_epsi = eps_LSR_noisy - delta_eps_LSR
                        y_train_cat[kk] = y_train_cat[kk]*(1-new_epsi) + new_epsi/num_classes

                elif LSRmode == 'WEIGHTS_CC' or LSRmode == 'WEIGHTS_CC+':

                    if LSRmode == 'WEIGHTS_CC':
                        # print('\n==distri_prior = False; WEIGHTS_CC')
                        perclass_weigths_cc = np.array(
                            [84, 20, 26, 32, 13, 44, 30, 74, 16, 56, 53, 74, 36, 29, 28, 56, 46, 35, 20, 61]) * 0.01

                    elif LSRmode == 'WEIGHTS_CC+':
                        # print('\n==distri_prior = False; WEIGHTS_CC+')
                        # define slithtly different
                        perclass_weigths_cc = np.array(
                            [84, 69, 65, 57, 50, 54, 30, 74, 16, 56, 53, 74, 63, 29, 28, 56, 54, 35, 27, 61]) * 0.01
                        # the idea: in some categories CC is misleading because despite some clips are Incorrect/OV,
                        # the true label is related and the acoustics is very similar, much more than to any other class in the vocab
                        # hence, we should use a lower CC. The new CCs:
                        # -bass guitar: 69
                        # -clapping: 65
                        # -coin: 57
                        # -crash: 50
                        # -dishes: 54
                        # -piano: 63
                        # -tearing: 54
                        # -wind: 27

                    # Define the known points
                    epsi_min = 0
                    epsi_max = eps_LSR_noisy
                    x_CC = np.array([0, 1], dtype=float)
                    y_epsi = np.array([epsi_max, epsi_min], dtype=float)

                    if LSRmapping == 'linear':
                        # Calculate the coefficients of the linear mapping
                        coeffs = np.polyfit(x_CC, y_epsi, 1)

                        # Find the polynomial, and apply it to the perclass_weigths_cc to get class-dependent epsilon
                        polynomial = np.poly1d(coeffs)
                        perclass_epsilon = polynomial(perclass_weigths_cc)

                    elif LSRmapping == 'quad':
                        # Calculate the coefficients of the quadratic mapping
                        coeffs = np.polyfit(x_CC ** 2, y_epsi, 1)

                        # apply coeffs to the input perclass_weigths_cc to get class-dependent epsilon
                        perclass_epsilon = coeffs[0]*(perclass_weigths_cc ** 2) + coeffs[1]

                    for kk in range(y_train_cat.shape[0]):
                        new_epsi = perclass_epsilon[np.nonzero(y_train_cat[kk])[0]]
                        y_train_cat[kk] = y_train_cat[kk]*(1-new_epsi) + new_epsi/num_classes


        elif distri_prior == 'noise':
            # watch distri_prior = noise. The energy to be spread across labels is not uniform, but in two levels
            # based on the amount of noise in the classes
            if not delta_eps_LSR:
                # watch delta_eps_LSR is None, hence all classes have the same epsilon for the ACTIVE class label
                # Inception implementation: max is not 1-epsi, but 1- epsi + epsi/n_classes
                # vip 1) define epsilon ALREADY DONE. in this case it is epsilon

                # vip 2) define distribution of (1-epsilon)
                distri_prior_noise = get_prior_distribution_noise(y_train_cat, num_classes, idx_noisy_classes,
                                                                  eps_LSR_noisy, delta_k)

                y_train_cat = y_train_cat*(1-eps_LSR_noisy) + distri_prior_noise
                # we are doing a re-distribution of the energy, the label vector still adds up to 1,
                # hence further processing in the losses wr2 origin holds

            else:
                # watch delta_eps_LSR is a float, hence some classes have higher/lower epsilon for the ACTIVE class label
                # depending on the amount of (more/less) noise. We use 2 levels of noisiness, hardcoded
                for kk in range(y_train_cat.shape[0]):

                    # vip 1) define epsilon with delta_eps_LSR
                    if np.nonzero(y_train_cat[kk])[0] in idx_noisy_classes:
                        # very noisy, increase epsi
                        new_epsi = eps_LSR_noisy + delta_eps_LSR
                    else:
                        # low noise, decrease epsi
                        new_epsi = eps_LSR_noisy - delta_eps_LSR

                    # vip 2) define distribution of (1-epsilon)
                    distri_prior_noise = get_prior_distribution_noise(y_train_cat[kk], num_classes, idx_noisy_classes, new_epsi, delta_k)

                    y_train_cat[kk] = y_train_cat[kk]*(1-new_epsi) + distri_prior_noise
                    # we are doing a re-distribution of the energy, the label vector still adds up to 1,


        elif distri_prior == 'unigram':
            # watch distri_prior = unigram. energy is distributed proportionally to the probability of the patches
            if not delta_eps_LSR:
                # watch delta_eps_LSR is None, hence all classes have the same epsilon for the ACTIVE class label
                # Inception implementation: max is not 1-epsi, but 1- epsi + epsi/n_classes
                # vip 1) define epsilon ALREADY DONE. in this case it is epsilon

                # vip 2) define distribution of (1-epsilon)
                distri_prior_unigram = get_prior_distribution_patch(patch_prior, eps_LSR_noisy)

                y_train_cat = y_train_cat*(1-eps_LSR_noisy) + distri_prior_unigram
                # we are doing a re-distribution of the energy, the label vector still adds up to 1,

            else:
                # watch delta_eps_LSR is a float, hence some classes have higher/lower epsilon for the ACTIVE class label
                # depending on the amount of (more/less) noise. We use 2 levels of noisiness, hardcoded
                for kk in range(y_train_cat.shape[0]):

                    # vip 1) define epsilon with delta_eps_LSR
                    if np.nonzero(y_train_cat[kk])[0] in idx_noisy_classes:
                        # very noisy, increase epsi
                        new_epsi = eps_LSR_noisy + delta_eps_LSR
                    else:
                        # low noise, decrease epsi
                        new_epsi = eps_LSR_noisy - delta_eps_LSR

                    # vip 2) define distribution of (1-epsilon)
                    distri_prior_unigram = get_prior_distribution_patch(patch_prior, new_epsi)

                    y_train_cat[kk] = y_train_cat[kk]*(1-new_epsi) + distri_prior_unigram
                    # we are doing a re-distribution of the energy, the label vector still adds up to 1

    elif mode == 'clean':
        # fix continue editing. todo lo que hiciste es para el noisy bencmark, hay que montarlo para modo origin.
        if not delta_eps_LSR:

            # Inception implementation: max is not 1-epsi, but 1- epsi + epsi/n_classes
            y_train_cat = y_train_cat*(1-epsilon_clean) + epsilon_clean/num_classes
            # vip we are doing a re-distribution of the energy, the label vector still adds up to 1,
            # hence further processing in the losses wr2 origin holds
        else:
            # not convex combination; allow free param for the energy of the non-labeled classes (does this make sense?)
            # am I breaking the rules here?
            y_train_cat = y_train_cat*(1-epsilon_clean) + energy_non_labeled/num_classes

    # fix mucho, why ha de ser siempre la misma distribution uniforme
    # porque siempre lo mismo para todas las muestras? se puede hacer algo aqui?
    # depende del subset de origen, puede que incluso de la clase?

    return y_train_cat


def get_one_hot(_all_labels, _patch_ids, n_classes):
    """
    get ground truth label (as one hot encoded vectors)
    given the entire dataset ground truth and a list of patch ids for the batch
    :param _all_labels:
    :param _patch_ids:
    :param n_classes:
    :return:
    """

    _y_int = np.empty((len(_patch_ids), 1), dtype='int')
    for tt in np.arange(len(_patch_ids)):
        _y_int[tt] = int(_all_labels[_patch_ids[tt]])
    _y_cat = to_categorical(_y_int, num_classes=n_classes)
    # ndarray (batch_size, n_classes)
    return _y_cat


def mixup(mode='intra', index=0, all_patch_indexes=None, batch_size=64, all_labels=None, all_features=None, alpha=0,
          n_classes=20, make_log=False, eps=0, clamp=0):
    """
    Apply mixup augmnentation. possibility to make it
    - intra batch
    - inter batch

    :param mode:
    :param index:
    :param all_patch_indexes:
    :param batch_size:
    :param all_labels:
    :param all_features:
    :param alpha:
    :param n_classes:
    :return:
    """

    if mode == 'intra':
        patch_ids = all_patch_indexes[index * batch_size:(index + 1) * batch_size]

        # fetch labels for the batch
        # ndarray (batch_size, n_classes)
        _y_cat1 = get_one_hot(all_labels, patch_ids, n_classes)

        # fetch features for the batch
        # (batch_size, time, freq)
        _features1 = all_features[patch_ids]

        # create randomized copies for both patches and labels (with correspondence)
        patch_ids_rand = permutation(patch_ids)
        _y_cat2 = get_one_hot(all_labels, patch_ids_rand, n_classes)
        _features2 = all_features[patch_ids_rand]
        # checked that y_cat and features are just randomized copies and there is feature-label correspondence

    elif mode == 'inter':

        patch_ids1 = all_patch_indexes[index * batch_size:(index + 1) * batch_size]
        patch_ids2 = all_patch_indexes[(index + 1) * batch_size:(index + 2) * batch_size]
        if index == int(np.floor(len(all_patch_indexes)/batch_size) - 1):
            # if final batch, do mixup with the very first (else, incomplete remaining batch)
            patch_ids2 = all_patch_indexes[0:batch_size]

        _y_cat1 = get_one_hot(all_labels, patch_ids1, n_classes)
        _features1 = all_features[patch_ids1]

        _y_cat2 = get_one_hot(all_labels, patch_ids2, n_classes)
        _features2 = all_features[patch_ids2]

    # apply mixup, can be optmized, this is more readable
    y_cat_out = np.zeros_like(_y_cat1)
    _features = np.zeros_like(_features1)

    lam = np.random.beta(alpha, alpha, batch_size)
    if clamp:
        # clamp values of lamda between [0.5:1]
        lam = np.maximum(lam, 0.5)

    for ii in range(batch_size):
        _features[ii] = lam[ii] * _features1[ii] + (1 - lam[ii]) * _features2[ii]
        y_cat_out[ii] = lam[ii] * _y_cat1[ii] + (1 - lam[ii]) * _y_cat2[ii]

        if make_log:
            _features[ii] = np.log10(_features[ii] + eps)

    # ojo con loas dimensiones de features. adjust format to input CNN
    # (batch_size, 1, time, freq) for channels_first
    features_out = _features[:, np.newaxis]

    return features_out, y_cat_out


def get_label_files(filelist=None, dire=None, suffix_in=None, suffix_out=None):
    """

    :param filelist: list of feature files (eg mel-spectrogram). only files (not path)
    :param dire:
    :param suffix_in: _mel
    :param suffix_out: _label
    :return: labels: ndarray of floats (nb_files_total, 1), meant for singly-labelled data
    labels is a list that follows the same order as the input param filelist (so there is correspondance features-label

    """

    nb_files_total = len(filelist)
    labels = np.zeros((nb_files_total, 1), dtype=np.float32)
    for f_id in range(nb_files_total):
        # load file containing the label and store in the ndarray
        labels[f_id] = utils_classif.load_tensor(in_path=os.path.join(dire, filelist[f_id].replace(suffix_in, suffix_out)))
    return labels


class DataGeneratorPatch(Sequence):
    """
    Reads data from disk and returns batches for training NNs.
    This is a version of DataGenerator tuned to allow T_F patch generation. Hence it mixes:
    -DataGenerator, based on wf, inheriting from Sequencer (grab the structure and i/o)
    -BatchGenerator, based on T_F patches (grab operations)


    adding here a scaler, with an option:
    -create and apply
    -load and apply
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    Parameters
    ----------
    feature_dir : string
        The path for the folder from where to load the input data to the network
    floatx: dtype
        Type of the arrays for the output
    """

    def __init__(self, feature_dir=None, file_list=None, params_learn=None, params_extract=None,
                 suffix_in='_mel', suffix_out='_label', floatx=np.float32, scaler=None):

        # TODO allow prepro here? preprocessing_fn = lambda x: x
        self.data_dir = feature_dir
        self.list_fnames = file_list
        # watch file_list is eg 12345_mel.data where 12345 is the fs_id
        # we dont load labels as they are stored in disk
        self.batch_size = params_learn.get('batch_size')
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.suffix_out = suffix_out
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))
        self.n_classes = int(params_learn.get('n_classes'))
        self.val_mode = False
        self.mode_last_patch = params_extract.get('mode_last_patch')

        # LSR
        self.LSR = params_learn.get('LSR')
        self.eps_LSR_noisy = params_learn.get('eps_LSR_noisy')
        self.delta_eps_LSR = params_learn.get('delta_eps_LSR')
        self.distri_prior = params_learn.get('distri_prior_LSR')
        self.delta_k = params_learn.get('delta_k_LSR')
        self.LSRmode = params_learn.get('LSRmode')
        self.LSRmapping = params_learn.get('LSRmapping')

        # mixup
        self.mixup = params_learn.get('mixup')
        self.mixup_mode = params_learn.get('mixup_mode')
        self.mixup_alpha = params_learn.get('mixup_alpha')
        self.mixup_make_log = params_learn.get('mixup_log')
        self.mixup_eps = params_extract.get('eps')
        self.mixup_clamp = params_learn.get('mixup_clamp')

        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        # - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        if feature_dir is not None:
            self.get_patches_features_labels(feature_dir, file_list)
            # at this point:
            # -all the TF patches from all the audio files are loaded into self.features
            # -their corresponding labels are in self.labels

            # vip let's count the patches per label (data imbalance)
            unique, counts = np.unique(self.labels, return_counts=True)
            patch_prior_asdict = dict(zip(unique, counts))
            # counts is already in the correct order, I think
            self.patch_prior = np.array(counts)
            self.patch_prior.shape = (1, self.n_classes)

            # standardize the data
            # self.features is a tensor with 3D. scaler expect 2D, we have to do the fix.
            # tensor, with the first dimension being the id of every T-F patch
            # self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if train set, create scaler, fit, transform, and save the scaler
            if scaler is None:
                self.scaler = StandardScaler()
                self.features2d = self.scaler.fit_transform(self.features2d)
                # this scaler will be used later on to scale val and test data
                # if the learning happens live, we keep it as attribute of the object
                # TODO save the scaler to avoid doing this unnecessarily

            else:
                self.val_mode = True
                # if we are in val or test set, load the training scaler as a param and transform
                self.features2d = scaler.transform(self.features2d)

            # after scaling in 2D, go back to tensor
            self.features = self.features2d.reshape(self.nb_inst_total, self.patch_len, self.feature_size)

        # but all the patches are contiguously ordered. shuffle them before making batches
        # this means we are shuffling them for the first time when we *initialize* the generator (before training)
        self.on_epoch_end()
        # on_epoch_end is triggered at init, and then at the end of each epoch to get new exploration order.
        # Shuffling order in which examples are fed, increases variability seen, hence robustness

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows, patches, or instances generated out of a given file
        """
        shape = utils_classif.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        if self.mode_last_patch == 'discard':
            # the last patch that is always incomplete is discarded
            if self.patch_len == 25 and self.patch_hop == 13 and file_frames == 51:
                num_instances_per_file = 3
            else:
                num_instances_per_file = np.maximum(1, int(np.ceil((file_frames - self.patch_len - 1) / self.patch_hop)))

        elif self.mode_last_patch == 'fill':
            # the last patch that is always incomplete will be filled with zeros or signal, to avoid discarding signal
            # hence we count one more patch
            if self.patch_len == 25 and self.patch_hop == 13 and file_frames == 51:
                num_instances_per_file = 3
            else:
                num_instances_per_file = np.maximum(1, 1 + int(np.ceil((file_frames - self.patch_len - 1) / self.patch_hop)))

        return num_instances_per_file

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils_classif.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features_labels(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        - shuffle them
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"
        print('Loading self.features...')
        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data') and
                          os.path.isfile(os.path.join(feature_dir, f.replace(self.suffix_in, self.suffix_out)))]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances (or T_F patches) per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_inst_total = self.nb_inst_cum[-1]

        # how many batches can we fit in the set
        self.nb_iterations = int(np.floor(self.nb_inst_total / self.batch_size))

        # this is not needed, as the inherited function from Sequencer does it for you
        # init iteration_step
        # self.iteration_step = -1

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features and labels
        # features is a tensor, with the first dimension being the id of every T-F patch
        self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
        # labels is just a column array, to store the label in float format
        self.labels = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)

        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk,
            # perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)

        # at this point: (imorove: not scalable)
        # -all the TF patches from all the audio files are loaded into self.features
        # -their corresponding labels are in self.labels

    def fetch_file_2_tensor(self, f_id):
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features

        mel_spec = utils_classif.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))
        label = utils_classif.load_tensor(in_path=os.path.join(self.feature_dir,
                                                       self.file_list[f_id].replace(self.suffix_in, self.suffix_out)))

        # indexes to store patches in self.features, according to the nb of instances from the file
        # (previously defined in get_num_instances_per_file)
        idx_start = self.nb_inst_cum[f_id]      # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]    # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):

            if idx == (idx_end - idx_start) - 1 and self.mode_last_patch == 'fill':
                # last patch and want to fill the incomplete patch
                tmp = mel_spec[start: start + self.patch_len]
                # I could leave it like this, since the rest are initialized to zeros,
                # but since I'm going to scale the features, artificial values are not cool.
                # So I fill it with the begining of the TF representation (circular shift)
                # watch: if the clip has 51 frames (extended to 1 second), in get_num_instances_per_file we allocated one patch
                # hence the 50 first frames is the last patch, so we get here, but tmp at this point has 50x128
                # so next line does nothing
                tmp = np.vstack((tmp, mel_spec[0: self.patch_len - tmp.shape[0]]))
                self.features[idx_start + idx] = tmp

            else:
                # rest of the cases
                self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]

            # update indexes
            start += self.patch_hop
            idx += 1

        self.labels[idx_start: idx_end] = label[0]

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.nb_iterations

    def __getitem__(self, index):
        """
        Generate one batch of data
        takes an index (batch number) and returns one batch of self.batch_size by calling
        __data_generation with the list of file names for the batch

        NOTE: the batch is composed ONLY of datapoints from the noisy subset
        if we do label smoothing, it is trivial

        :param index:
        :return:
        """

        # index is taken care of by the Sequencer inherited
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # fetch labels for the batch
        # y = [self.labels[k] for k in indexes]
        y_int = np.empty((self.batch_size, 1), dtype='int')
        for tt in np.arange(self.batch_size):
            y_int[tt] = int(self.labels[indexes[tt]])
        y_cat = to_categorical(y_int, num_classes=self.n_classes)

        if self.LSR and self.val_mode is False:
        # if self.LSR:
            # vip add label-smoothing fix improve
            # epsilon = 0.05 / 0.3
            y_cat = label_smoothing(y_cat,
                                    eps_LSR_noisy=self.eps_LSR_noisy,
                                    epsilon_clean=0,
                                    delta_eps_LSR=self.delta_eps_LSR,
                                    num_classes=self.n_classes,
                                    mode='all',
                                    distri_prior=self.distri_prior,
                                    delta_k=self.delta_k,
                                    patch_prior=self.patch_prior,
                                    LSRmode=self.LSRmode,
                                    LSRmapping=self.LSRmapping
                                    )

        # fetch features for the batch and adjust format to input CNN
        # (batch_size, 1, time, freq) for channels_first
        features = self.features[indexes, np.newaxis]

        if self.mixup and self.val_mode is False:
            features, y_cat = mixup(mode=self.mixup_mode,
                                    index=index,
                                    all_patch_indexes=self.indexes,
                                    batch_size=self.batch_size,
                                    all_labels=self.labels,
                                    all_features=self.features,
                                    n_classes=self.n_classes,
                                    alpha=self.mixup_alpha,
                                    make_log=self.mixup_make_log,
                                    eps=self.mixup_eps,
                                    clamp=self.mixup_clamp
                                    )
        elif self.mixup and self.val_mode and self.mixup_make_log:
            # mel spectrograms have no log. for the training subset we apply it after mixing up
            # but in the val subset there is no mixup, but we need to apply the log on the fly
            for ii in range(self.batch_size):
                features[ii, 0, :] = np.log10(features[ii, 0, :] + self.mixup_eps)

        return features, y_cat

    def on_epoch_end(self):
        # shuffle data between epochs
        self.indexes = np.random.permutation(self.nb_inst_total)


class PatchGeneratorPerFile(object):
    """
    Reads whole T_F representations from disk,
    and stores T_F patches *for a given entire file* in a tensor
    typically for prediction on a test set

    Should allow standardization of the tensor, with a scaler

    Parameters
    ----------

    """

    def __init__(self, feature_dir=None, file_list=None, params_extract=None,
                 suffix_in='_mel', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))
        self.mode_last_patch = params_extract.get('mode_last_patch')

        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        if feature_dir is not None:
            self.get_patches_features(feature_dir, file_list)
            # at this point:
            # -all the TF patches from all the audio files are loaded into self.features
            # -sorted by file, ie, all the patches are contiguously ordered.

            # standardize the data: assuming this is used for inference
            # self.features is a tensor with 3D. scaler expect 2D, we have to do the fix.
            # tensor, with the first dimension being the id of every T-F patch
            # self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if we are in val or test subset, load the training scaler as a param and transform
            self.features2d = scaler.transform(self.features2d)

            # go back to 3D tensor
            self.features = self.features2d.reshape(self.nb_patch_total, self.patch_len, self.feature_size)

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows, patches, or instances generated out of a given file
        """
        shape = utils_classif.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        if self.mode_last_patch == 'discard':
            # the last patch that is always incomplete is discarded
            if self.patch_len == 25 and self.patch_hop == 13 and file_frames == 51:
                num_instances_per_file = 3
            else:
                num_instances_per_file = np.maximum(1, int(np.ceil((file_frames - self.patch_len - 1) / self.patch_hop)))

        elif self.mode_last_patch == 'fill':
            # the last patch that is always incomplete will be filled with zeros or signal, to avoid discarding signal
            # hence we count one more patch
            if self.patch_len == 25 and self.patch_hop == 13 and file_frames == 51:
                num_instances_per_file = 3
            else:
                num_instances_per_file = np.maximum(1, 1 + int(np.ceil((file_frames - self.patch_len - 1) / self.patch_hop)))

        return num_instances_per_file

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils_classif.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"

        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data')]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_patch_total = self.nb_inst_cum[-1]

        # init current file, to keep track of the file yielded for prediction
        self.current_f_idx = 0

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features
        # features is a tensor, with the first dimension being the id of every T-F patch
        self.features = np.zeros((self.nb_patch_total, self.patch_len, self.feature_size), dtype=self.floatx)

        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk,
            # perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)
        # at this point:
        # -all the TF patches from all the audio files are loaded into self.features, and sorted

    def fetch_file_2_tensor(self, f_id):
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features

        mel_spec = utils_classif.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))

        # indexes to store patches in self.features, according to the nb of instances from the file
        # (previously defined in get_num_instances_per_file)
        idx_start = self.nb_inst_cum[f_id]      # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]    # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):

            if idx == (idx_end - idx_start) - 1 and self.mode_last_patch == 'fill':
                # last patch and want to fill the incomplete patch
                tmp = mel_spec[start: start + self.patch_len]
                # I could leave it like this, since the rest are initialized to zeros,
                # but since I'm going to scale the features, artificial values are not cool.
                # So I fill it with the begining of the TF representation (circular shift)
                # watch: if the clip has 51 frames (extended to 1 second), in get_num_instances_per_file we allocated one patch
                # hence the 50 first frames is the last patch, so we get here, but tmp at this point has 50x128
                # so next line does nothing
                tmp = np.vstack((tmp, mel_spec[0: self.patch_len - tmp.shape[0]]))
                self.features[idx_start + idx] = tmp

            else:
                # rest of the cases
                self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]

            # update indexes
            start += self.patch_hop
            idx += 1

    def get_patches_file(self):
        """
        Returns all the patches for one single audio clip
        """

        self.current_f_idx += 1
        # ranges form 1 to self.nb_files (ignores 0)
        assert self.current_f_idx <= self.nb_files, 'All the test files have been dispatched'

        # fetch features in the batch and adjust format to input CNN
        # (nb_patches_per_file, 1, time, freq)
        features = self.features[self.nb_inst_cum[self.current_f_idx-1]: self.nb_inst_cum[self.current_f_idx], np.newaxis]
        return features



"""
new generator that allows to create one-hot vectors carrying flags in the hot label
ie 100. instead of 1.
this is used in the loss functions to distinguish patches coming from noisy or clean set
"""


class DataGeneratorPatchOrigin(Sequence):
    """
    Reads data from disk and returns batches for training NNs.
    This is a version of DataGenerator tuned to allow T_F patch generation. Hence it mixes:
    -DataGenerator, based on wf, inheriting from Sequencer (grab the structure and i/o)
    -BatchGenerator, based on T_F patches (grab operations)


    adding here a scaler, with an option:
    -create and apply
    -load and apply
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    Parameters
    ----------
    feature_dir : string
        The path for the folder from where to load the input data to the network
    floatx: dtype
        Type of the arrays for the output
    """

    def __init__(self, feature_dir=None, file_list=None, params_learn=None, params_extract=None,
                 suffix_in='_mel', suffix_out='_label', floatx=np.float32, scaler=None):

        # TODO allow prepro here? preprocessing_fn = lambda x: x
        self.data_dir = feature_dir
        self.list_fnames = file_list
        # we dont load labels as they are stored in disk
        self.batch_size = params_learn.get('batch_size')
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.suffix_out = suffix_out
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))
        self.n_classes = params_learn.get('n_classes')
        self.noisy_ids = params_learn.get('noisy_ids')
        # LSR
        self.LSR = params_learn.get('LSR')
        self.eps_LSR_noisy = params_learn.get('eps_LSR_noisy')
        self.eps_LSR_clean = params_learn.get('eps_LSR_clean')
        self.delta_eps_LSR = params_learn.get('delta_eps_LSR')

        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        # - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        if feature_dir is not None:
            self.get_patches_features_labels(feature_dir, file_list)
            # at this point:
            # -all the TF patches from all the audio files are loaded into self.features
            # -their corresponding labels are in self.labels

            # standardize the data
            # self.features is a tensor with 3D. scaler expect 2D, we have to do the fix.
            # tensor, with the first dimension being the id of every T-F patch
            # self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if train set, create scaler, fit, transform, and save the scaler
            if scaler is None:
                self.scaler = StandardScaler()
                self.features2d = self.scaler.fit_transform(self.features2d)
                # this scaler will be used later on to scale val and test data
                # if the learning happens live, we keep it as attribute of the object
                # TODO save the scaler to avoid doing this unnecessarily

            else:
                self.val_mode = True
                # if we are in val or test set, load the training scaler as a param and transform
                self.features2d = scaler.transform(self.features2d)

            # after scaling in 2D, go back to tensor
            self.features = self.features2d.reshape(self.nb_inst_total, self.patch_len, self.feature_size)

        # but all the patches are contiguously ordered. shuffle them before making batches
        # this means we are shuffling them for the first time when we *initialize* the generator (before training)
        self.on_epoch_end()

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows, patches, or instances generated out of a given file
        """
        shape = utils_classif.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils_classif.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features_labels(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        - shuffle them
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"
        print('Loading self.features...')
        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data') and
                          os.path.isfile(os.path.join(feature_dir, f.replace(self.suffix_in, self.suffix_out)))]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances (or T_F patches) per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_inst_total = self.nb_inst_cum[-1]

        # how many batches can we fit in the set
        self.nb_iterations = int(np.floor(self.nb_inst_total / self.batch_size))

        # this is not needed, as the inherited function from Sequencer does it for you
        # init iteration_step
        # self.iteration_step = -1

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features and labels
        # features is a tensor, with the first dimension being the id of every T-F patch
        self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
        # labels is just a column array, to store the label in float format
        self.labels = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)
        # analogous column vector to flag patches coming from noisy subset of train data
        # init to 0. Only 1 if they come from noisy subset
        self.noisy_patches = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)

        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk,
            # perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)

        # at this point: (imorove: not scalable)
        # -all the TF patches from all the audio files are loaded into self.features
        # -their corresponding labels are in self.labels

    def fetch_file_2_tensor(self, f_id):
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features

        mel_spec = utils_classif.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))
        label = utils_classif.load_tensor(in_path=os.path.join(self.feature_dir,
                                                       self.file_list[f_id].replace(self.suffix_in, self.suffix_out)))

        # indexes to store patches in self.features, according to the nb of instances from the file
        idx_start = self.nb_inst_cum[f_id]      # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]    # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]
            # update indexes
            start += self.patch_hop
            idx += 1

        self.labels[idx_start: idx_end] = label[0]

        # self.file_list[f_id].split('_')[0] is the fs id in str
        if int(self.file_list[f_id].split('_')[0]) in self.noisy_ids:
            # if the clip comes from noisy subset, flag to 1 all its patches
            self.noisy_patches[idx_start: idx_end] = 1

    def __len__(self):
        return self.nb_iterations

    def __getitem__(self, index):
        """
        takes an index (batch number) and returns one batch of self.batch_size by calling
        __data_generation with the list of file names for the batch
        :param index:
        :return:
        """
        # vip index is taken care of by the inherited Sequencer
        # indexes is a ndarray of (self.batch_size,) eg (64,) indexes of the TF patches to return
        # these indexes are in [0: self.nb_inst_total]
        # they've been previously shuffled in every epoch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # fetch labels for the batch
        # y = [self.labels[k] for k in indexes]
        y_int = np.empty((self.batch_size, 1), dtype='int')
        for tt in np.arange(self.batch_size):
            y_int[tt] = int(self.labels[indexes[tt]])

        # one-hot vectors, into a matrix batch_size, n_classes
        y_cat = to_categorical(y_int, num_classes=self.n_classes)

        # vip apply label smoothing selectively, based on origin
        # watch important to do this before applying the flagging factor! else it does not work
        if self.LSR and self.val_mode is False:
            for tt in np.arange(self.batch_size):
                if self.noisy_patches[indexes[tt]] == 1:
                    # patch coming from clip in the NOISY subset
                    # epsilon = 0.05 / 0.3
                    y_cat[tt] = label_smoothing(y_cat[tt],
                                                eps_LSR_noisy=self.eps_LSR_noisy,
                                                epsilon_clean=0,
                                                delta_eps_LSR=self.delta_eps_LSR,
                                                num_classes=self.n_classes,
                                                mode='noisy')
                else:
                    # patch coming from clip in the CLEAN subset
                    y_cat[tt] = label_smoothing(y_cat[tt],
                                                eps_LSR_noisy=0,
                                                epsilon_clean=self.eps_LSR_clean,
                                                delta_eps_LSR=self.delta_eps_LSR,
                                                num_classes=self.n_classes,
                                                mode='clean')

        # vip tune the one-hot vectors of the patches coming from clips in the noisy subset
        for tt in np.arange(self.batch_size):
            if self.noisy_patches[indexes[tt]] == 1:
                y_cat[tt] *= 100

        # fetch features for the batch and adjust format to input CNN
        # (batch_size, 1, time, freq) for channels_first
        features = self.features[indexes, np.newaxis]
        return features, y_cat

    def on_epoch_end(self):
        # shuffle data between epochs
        self.indexes = np.random.permutation(self.nb_inst_total)
