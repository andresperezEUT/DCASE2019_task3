
# basic
import numpy as np
import scipy
import librosa
import soundfile


"""
Some of these functions have been inspired on the DCASE UTIL framework by Toni Heittola
https://dcase-repo.github.io/dcase_util/

"""


def load_audio_file(file_path, input_fixed_length=0, params_extract=None):
    """
    TO be IMPROVED
    -load entire audio file
    -resample if needed
    -normalize it
    -reshape

    -input_fixed_length is only used in case we cannot read the file (and generates a vector of ones with that length)
    """

    data, source_fs = soundfile.read(file=file_path)
    data = data.T

    # Resample if the source_fs is different from expected
    if params_extract.get('fs') != source_fs:
        data = librosa.core.resample(data, source_fs, params_extract.get('fs'))
        print('Resampling to %d: %s' % (params_extract.get('fs'), file_path))

    if len(data) > 0:
        data = get_normalized_audio(data)
    else:
        # 3 files are corrupted in the test set. They belong to the padding group (not used for evaluation)
        data = np.ones((input_fixed_length, 1))
        print('File corrupted. Could not open: %s' % file_path)

    # careful with the shape
    data = np.reshape(data, [-1, 1])
    return data


def modify_file_variable_length(data=None, input_fixed_length=0, params_extract=None):
    """
    data is the entire audio file loaded, with proper shape
    -depending on the loading mode (in params_extract)
    --FIX: if sound is short, replicate sound to fill up to input_fixed_length
           if sound is too long, grab only a (random) slice of size  input_fixed_length
    --VARUP: short sounds get replicated to fill up to input_fixed_length
    --VARFULL: this function is a by pass (hence full length is considered)
    :return:
    """

    # NOTE: data is a column vector
    if params_extract.get('load_mode') == 'fix' or params_extract.get('load_mode') == 'varup':
        # deal with short sounds
        if len(data) < input_fixed_length:
            # if file shorter than input_length, replicate the sound to reach the input_fixed_length
            nb_replicas = int(np.ceil(input_fixed_length / len(data)))
            # replicate according to column
            data_rep = np.tile(data, (nb_replicas, 1))
            data = data_rep[:input_fixed_length]

    if params_extract.get('load_mode') == 'fix':
        # deal with long sounds
        if len(data) > input_fixed_length:
            # if file longer than input_length, grab input_length from a random slice of the file
            max_offset = len(data) - input_fixed_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_fixed_length + offset)]

    return data


"""----------------------------------mine"""
# ------------------------------------------------


# def load_audio_file(audio_file, params):
#     """Load audio using AudioFile class
#
#     Parameters
#     ----------
#     audio_file : str
#     params : dict
#
#     Returns
#     -------
#     numpy.ndarray
#         Audio data
#
#     fs : int
#         Sampling frequency
#
#     """
#
#     # Collect parameters
#     # mono = params['mono']
#     # diff = params['diff']
#     # fs = params['fs']
#
#     # normalize_audio = False
#     # if 'normalize_audio' in params:
#     #     normalize_audio = params['normalize_audio']
#
#     # y, fs = AudioFile().load(filename=audio_file, mono=mono, diff=diff, fs=fs)
#
#     # y, source_fs = soundfile.read(file=audio_file)
#     # y = y.T
#
#     # # Down-mix audio
#     # if mono and len(y.shape) > 1:
#     #     # we must output one channel only
#     #     if not diff:
#     #         # if diff = False, compute mono signal as usual
#     #         y = np.mean(y, axis=0)
#     #     else:
#     #         # if diff = True, compute difference between L and R and center it by removing DC
#     #         left_right = y[0, :] - y[1, :]
#     #         y = left_right - np.mean(left_right)
#
#     # # Resample if the source_fs is different from the indicated one
#     # if fs != source_fs:
#     #     import librosa
#     #     y = librosa.core.resample(y, source_fs, fs)
#     #
#     # y = np.reshape(y, [1, -1])
#     #
#     # # Normalize audio
#     # if normalize_audio:
#     #     for channel in range(0, y.shape[0]):
#     #         y[channel] = get_normalized_audio(y[channel])
#     #
#     # return y, fs


def get_normalized_audio(y, head_room=0.005):
    """Normalize audio

    Parameters
    ----------
    y : numpy.ndarray
        Audio data
    head_room : float
        Head room

    Returns
    -------
    numpy.ndarray
        Audio data

    """

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value


def get_window_function(N, window_type='hamming_asymmetric'):
    """Window function

    Parameters
    ----------
    N : int
        window length

    window_type : str
        window type
        (Default value='hamming_asymmetric')
    Raises
    ------
    ValueError:
        Unknown window type

    Returns
    -------
        window function : array
    """

    # Windowing function
    if window_type == 'hamming_asymmetric':
        return scipy.signal.hamming(N, sym=False)
    elif window_type == 'hamming_symmetric':
        return scipy.signal.hamming(N, sym=True)
    elif window_type == 'hann_asymmetric':
        return scipy.signal.hann(N, sym=False)
    elif window_type == 'hann_symmetric':
        return scipy.signal.hann(N, sym=True)
    else:
        message = 'Unknown window type [{window_type}]'.format(
            window_type=window_type
        )
        raise ValueError(message)


def get_mel_spectrogram(audio, params_extract=None):
    """Mel-band energies

    Parameters
    ----------
    audio : numpy.ndarray
        Audio data.
    params : dict
        Parameters.

    Returns
    -------
    feature_matrix : numpy.ndarray
        (log-scaled) mel spectrogram energies per audio channel

    """
    # make sure rows are channels and columns the samples
    audio = audio.reshape([1, -1])

    window = get_window_function(N=params_extract.get('win_length_samples'),
                                 window_type=params_extract.get('window'))

    mel_basis = librosa.filters.mel(sr=params_extract.get('fs'),
                                    n_fft=params_extract.get('n_fft'),
                                    n_mels=params_extract.get('n_mels'),
                                    fmin=params_extract.get('fmin'),
                                    fmax=params_extract.get('fmax'),
                                    htk=params_extract.get('htk'),
                                    norm=params_extract.get('mel_basis_unit'))

    if params_extract.get('normalize_mel_bands'):
        mel_basis /= np.max(mel_basis, axis=-1)[:, None]

    # init mel_spectrogram expressed as features: row x col = frames x mel_bands = 0 x mel_bands (to concatenate axis=0)
    feature_matrix = np.empty((0, params_extract.get('n_mels')))
    for channel in range(0, audio.shape[0]):
        spectrogram = get_spectrogram(
            y=audio[channel, :],
            n_fft=params_extract.get('n_fft'),
            win_length_samples=params_extract.get('win_length_samples'),
            hop_length_samples=params_extract.get('hop_length_samples'),
            spectrogram_type=params_extract.get('spectrogram_type') if 'spectrogram_type' in params_extract else 'magnitude',
            center=True,
            window=window,
            params_extract=params_extract
        )

        mel_spectrogram = np.dot(mel_basis, spectrogram)
        mel_spectrogram = mel_spectrogram.T
        # at this point we have row x col = time x freq = frames x mel_bands

        if params_extract.get('log'):
            mel_spectrogram = np.log10(mel_spectrogram + params_extract.get('eps'))

        # if there is more than one channel, we simply concatenate spectrograms into the ndarray
        # no explicit separation. (we could make a list of lists)
        feature_matrix = np.append(feature_matrix, mel_spectrogram, axis=0)

    return feature_matrix


def get_spectrogram(y,
                    n_fft=1024,
                    win_length_samples=0.04,
                    hop_length_samples=0.02,
                    window=scipy.signal.hamming(1024, sym=False),
                    center=True,
                    spectrogram_type='magnitude',
                    params_extract=None):

    """Spectrogram

    Parameters
    ----------
    y : numpy.ndarray
        Audio data
    n_fft : int
        FFT size
        Default value "1024"
    win_length_samples : float
        Window length in seconds
        Default value "0.04"
    hop_length_samples : float
        Hop length in seconds
        Default value "0.02"
    window : array
        Window function
        Default value "scipy.signal.hamming(1024, sym=False)"
    center : bool
        If true, input signal is padded so to the frame is centered at hop length
        Default value "True"
    spectrogram_type : str
        Type of spectrogram "magnitude" or "power"
        Default value "magnitude"

    Returns
    -------
    np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix

    """

    if spectrogram_type == 'magnitude':
        return np.abs(librosa.stft(y + params_extract.get('eps'),
                                   n_fft=n_fft,
                                   win_length=win_length_samples,
                                   hop_length=hop_length_samples,
                                   center=center,
                                   window=window))
    elif spectrogram_type == 'power':
        return np.abs(librosa.stft(y + params_extract.get('eps'),
                                   n_fft=n_fft,
                                   win_length=win_length_samples,
                                   hop_length=hop_length_samples,
                                   center=center,
                                   window=window)) ** 2
    else:
        message = 'Unknown spectrum type [{spectrogram_type}]'.format(
            spectrogram_type=spectrogram_type
        )
        raise ValueError(message)
