import os.path
import soundfile as sf
import numpy as np
import csv
import sys
pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
import parametric_spatial_audio_processing as psa
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import scipy.stats

plt.style.use('default')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FOLDERS

# Development data path
parent_folder_path = '/Volumes/Dinge/DCASE2019_subset'
data_folder_path = os.path.join(parent_folder_path, 'foa_dev')
# metadata_folder_path = os.path.join(parent_folder_path, 'metadata_dev')
result_folder_path = '/Users/andres.perez/source/DCASE2019/results/subset'
result_file_extension = '.csv'



# metadata_files = os.listdir(metadata_folder_path)
metadata_extension = '.csv'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CLASSES

class Groundtruth:
    def __init__(self):
        self.type = []
        self.onset = []
        self.offset = []
        self.ele = []
        self.azi = []
        self.dist = []

    def add(self, row):
        self.type.append(row[0])
        self.onset.append(row[1])
        self.offset.append(row[2])
        self.ele.append(row[3])
        self.azi.append(row[4])
        self.dist.append(row[5])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ANALYSIS PARAMETERS

window_size = 256
window_overlap = window_size / 2
nfft = window_size

energy_th = 1e-6

fmin = 125
fmax = 8000
dt = 4
directivity_th = 0.5
num_min_directive_bins = 10


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PROCESSING

kmeans_azi = None

# TODO: reimplement it more fancy...
def discard_nans(array):
    new = []
    for el in array:
        if not np.isnan(el):
            new.append(el)
    return np.asarray(new)

def rad2deg(rad):
    return rad*360/(2*np.pi)

def circmedian(angs):
    # from https://github.com/scipy/scipy/issues/6644
    # Radians!
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    pdists = np.abs(pdists).sum(1)
    return angs[np.argmin(pdists)]

def process(data, sr):

    # Assert first order ambisonics and dimensionality order
    num_frames = np.shape(data)[0]
    num_channels = np.shape(data)[1]
    assert num_channels == 4


    start_frame = 0
    # end_frame = num_frames
    end_frame = np.shape(data)[0]

    x = psa.Signal(data[start_frame:end_frame].T, sr, 'acn', 'n3d')
    X = psa.Stft.fromSignal(x,
                            window_size=window_size,
                            window_overlap=window_overlap,
                            nfft=nfft).limit_bands(fmin, fmax)

    e = psa.compute_energy_density(x)
    psa.plot_signal(e, y_scale='log')

    # env = psa.compute_signal_envelope(e, windowsize=1024)
    # plt.plot(e.data[0])
    # plt.plot(env.data[0])
    # plt.hlines(energy_density_th,0,x.get_num_frames())
    # plt.yscale('log')
    # plt.show()

    ################
    N = X.get_num_time_bins()
    K = X.get_num_frequency_bins()

    # energy_n = np.zeros(N)
    # for k in range(K):
    #     for n in range(N):
    #         energy_n[n] = np.sum(np.power(np.abs(X.data[:,:,n]),2))
    # energy_smooth = scipy.signal.savgol_filter(energy_n, 257, 5)
    # plt.figure()
    # plt.plot(energy_n)
    # plt.plot(energy_smooth)
    # plt.yscale('log')
    # plt.grid()
    # plt.show()
    #
    # activity_bins = []
    # onset_samples = []
    # offset_samples = []
    # for n in x.get_num_frames():
    #     if energy_smooth[n] > energy_th:
    #         activity_bins.append()

    psa.plot_magnitude_spectrogram(X)

    doa = psa.compute_DOA(X)

    directivity = X.compute_ksi2_re(dt=dt)
    directivity_mask = directivity.compute_mask(th=directivity_th)
    doa_th = doa.apply_mask(directivity_mask)

    psa.plot_doa(doa, title=str(dt))
    psa.plot_directivity(directivity, title=str(dt))
    psa.plot_mask(directivity_mask, title=str(dt))
    psa.plot_doa(doa_th, title=str(dt))
    plt.show()

    N = X.get_num_time_bins()
    K = X.get_num_frequency_bins()


    ## Compute position statistics
    activity_bins = []
    position_mean = []
    position_std = []
    position_median = []

    for n in range(N):
        azi = discard_nans(doa_th.data[0,:,n]).reshape(-1,1)
        ele = discard_nans(doa_th.data[1,:,n]).reshape(-1,1)

        if np.size(azi) < num_min_directive_bins:
            # Empty! not enough suitable doa values
            pass
        else:
            activity_bins.append(n)

            mean_azi = rad2deg(scipy.stats.circmean(azi, high=np.pi, low=-np.pi))
            mean_ele = rad2deg(np.mean(ele))
            position_mean.append([mean_azi, mean_ele])

            std_azi = rad2deg(scipy.stats.circstd(azi, high=np.pi, low=-np.pi))
            std_ele = rad2deg(np.std(ele))
            position_std.append([std_azi, std_ele])

            median_azi = rad2deg(circmedian(azi))
            median_ele = rad2deg(np.median(ele))
            position_median.append([median_azi, median_ele])


    ## Find contiguous regions

    audio_event_min_separation = 100  # bins @256 window size
    audio_event_min_duration = 50 # bins @256 window size
    onsets = []
    offsets = []

    for n in range(N):
        if n in activity_bins:
            index_of_current_active_bin = activity_bins.index(n)

            ## Find onsets
            if index_of_current_active_bin == 0:
                # This is the first active bin, so probably it's an onset
                onsets.append(n)
            else:
                last_active_bin = activity_bins[index_of_current_active_bin - 1]
                if (n - last_active_bin) > audio_event_min_separation:
                    onsets.append(n)

            ## Find offsets
            if index_of_current_active_bin == len(activity_bins)-1:
                # This is the last active bin, so probably it's an offset
                offsets.append(n)
            else:
                next_active_bin = activity_bins[index_of_current_active_bin + 1]
                if (next_active_bin - n) > audio_event_min_separation:
                    offsets.append(n)

    assert len(onsets) == len(offsets)

    # Take only the events larger than the minimum duration (remove spureous and transients)
    postprocessed_onsets = []
    postprocessed_offsets = []
    event_lenghts = np.asarray(offsets) - np.asarray(onsets)
    for event_idx, event_length in enumerate(event_lenghts):
        if event_length > audio_event_min_duration:
            postprocessed_onsets.append(onsets[event_idx])
            postprocessed_offsets.append(offsets[event_idx])

    assert len(postprocessed_onsets) == len(postprocessed_offsets)


    ## Plot the nice graphs over time
    ymin_azi = -200
    ymax_azi = 200
    ymin_ele = -100
    ymax_ele = 100
    with plt.style.context(('seaborn-whitegrid')):
        plt.figure(figsize=plt.figaspect(1/2.))
        plt.suptitle('azimuth')
        plt.errorbar(activity_bins, np.asarray(position_mean)[:,0], yerr=np.asarray(position_std)[:,0],  fmt='o', color='black',
                 ecolor='lightgray', elinewidth=1, capsize=0, markersize=1)
        plt.ylim(ymin_azi, ymax_azi)
        plt.errorbar(activity_bins, np.asarray(position_median)[:,0], fmt='o', color='red', markersize=1)
        plt.vlines(postprocessed_onsets, ymin_azi, ymax_azi, linestyles='dashed', colors='k')
        plt.vlines(postprocessed_offsets, ymin_azi, ymax_azi, linestyles='dashed', colors='b')
        plt.show()

        plt.figure(figsize=plt.figaspect(1 / 2.))
        plt.suptitle('elevation')
        plt.errorbar(activity_bins, np.asarray(position_mean)[:,1], yerr=np.asarray(position_std)[:,1], fmt='o', color='black',
                 ecolor='lightgray', elinewidth=1, capsize=0, markersize=1)
        plt.ylim(ymin_ele, ymax_ele)
        plt.errorbar(activity_bins, np.asarray(position_median)[:, 1], fmt='o', color='red', markersize=1)
        plt.vlines(postprocessed_onsets, ymin_azi, ymax_azi, linestyles='dashed', colors='k')
        plt.vlines(postprocessed_offsets, ymin_azi, ymax_azi, linestyles='dashed', colors='b')
        plt.show()



    ## Bounce data to output format [frame_id, class_id, azimuth, elevation]
    #   For the moment, let's stick with the window size of the analysis,
    #   and we will perform the conversion to the required hopsize later

    # result = [bin, class_id, azi, ele] with likely repeated bin instances
    result = []
    class_id = 0 #whatever for the moment
    for bin_idx, bin in enumerate(activity_bins):
        azi = position_mean[bin_idx][0]
        ele = position_mean[bin_idx][1]
        result.append( [bin, class_id, azi, ele] )

    ## Perform the window transformation by averaging within new bin
    ## TODO: assert our bins are smaller than required ones

    current_window_hop = (window_size-window_overlap) / float(sr)
    required_window_hop = 0.02
    window_factor = required_window_hop/current_window_hop

    # Since bins are ordered (at least they should), we can optimise that a little bit
    last_bin = -1
    # result_quantized = [bin, [class_id, azi, ele],[class_id, azi, ele]... ] without repeated bin instances
    result_quantized = []
    for row in result:
        bin = row[0]
        new_bin = int(np.floor(bin/window_factor))
        if new_bin == last_bin:
            result_quantized[-1].append( [ row[1], row[2], row[3] ] )
        else:
            result_quantized.append([new_bin, [row[1], row[2], row[3]]])
        last_bin = new_bin

    # Let's average them within the quantized_bin
    # TODO: in the future, probably discriminate here between multiple sources based on std or something...

    result_averaged_dict = {}
    for row in result_quantized:
        bin = row[0]
        label = 0 #TODO

        azis = np.asarray(row[1:])[:,1]
        azi = scipy.stats.circmean(azis, high=180, low=-180)
        eles = np.asarray(row[1:])[:,2]
        ele = np.mean(eles)

        result_averaged_dict[bin] = [label, azi, ele]

    return result_averaged_dict











    # ## Plot histograms
    #
    # inertia_ratio_th = 2.
    #
    # # num_bins = [360, 180] # 1 degree
    # for event_idx in range(len(postprocessed_offsets)):
    #
    #     start_idx = postprocessed_onsets[event_idx]
    #     end_idx = postprocessed_offsets[event_idx]
    #
    #     azis = discard_nans(doa_th.data[0, :, start_idx:end_idx].ravel())
    #     eles = discard_nans(doa_th.data[1, :, start_idx:end_idx].ravel())
    #
    #     # Kmeans
    #     kmeans_data = np.asarray([azis,eles]).T
    #     inertia = []
    #     for k in [1,2]:
    #         kmeans = KMeans(n_clusters=k, random_state=0).fit(kmeans_data)
    #         inertia.append(kmeans.inertia_)
    #
    #     inertia_ratio = inertia[0]/inertia[1]
    #     # print(event_idx, inertia_ratio)
    #
    #     if inertia_ratio > inertia_ratio_th:
    #         # There are two clusters
    #         pass
    #     else:
    #         # Just one cluster
    #         pass
    #
    #     # # Histogram
    #     # H, xedges, yedges = np.histogram2d(azis, eles, num_bins)
    #     # plt.figure(figsize=plt.figaspect(1/2.))
    #     # plt.suptitle(str(start_idx)+','+str(end_idx))
    #     # plt.pcolormesh(H.T)
    #     # plt.show()


    # # Doa Variance
    #
    # vicinity_radius = 1
    # std = np.zeros((2, K, N))
    #
    # for k in range(vicinity_radius, K-vicinity_radius):
    #     for n in range(vicinity_radius, N-vicinity_radius):
    #
    #         range_k = range(k-vicinity_radius, k+vicinity_radius+1)
    #         range_n = range(n-vicinity_radius, n+vicinity_radius+1)
    #
    #         std[0, k, n] = scipy.stats.circstd(doa.data[0, range_k, range_n], high=np.pi, low=-np.pi)
    #         std[1, k, n] = np.std(doa.data[1, range_k, range_n])
    #
    # # Edges: copy values
    # for m in range(2):
    #     for k in range(0, vicinity_radius):
    #         std[m, k, :] = std[m, k+vicinity_radius, :]
    #     for k in range(K-vicinity_radius, K):
    #         std[m, k, :] = std[m, K-vicinity_radius-1, :]
    #     for n in range(0, vicinity_radius):
    #         std[m, :, n] = std[m, :, n + vicinity_radius]
    #     for n in range(N - vicinity_radius, N):
    #         std[m, :, n] = std[m, :, N - vicinity_radius - 1,]
    #
    # plt.figure(figsize=plt.figaspect(1 / 2.))
    # plt.subplot(211)
    # plt.pcolormesh(std[0])
    # plt.colorbar()
    # plt.subplot(212)
    # plt.pcolormesh(std[1])
    # plt.colorbar()
    # plt.show()
    #
    # doa_std = psa.Stft(doa.t, doa.f, std, doa.sample_rate)
    #
    # th_local = doa_std.compute_threshold_local(block_size=51, method='mean')



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAIN

print('-------------- PROCESSING FILES --------------')
print('                                              ')
print('Folder path: ' + data_folder_path              )
print('                                              ')

# Check if results destination folder exists, and create it in turn
if not os.path.exists(result_folder_path):
    os.mkdir(result_folder_path)

# Iterate over all audio files
for audio_file in os.listdir(data_folder_path):
# for audio_file in ['split1_ir0_ov1_1.wav']:
# for audio_file in ['split2_ir1_ov2_35.wav']:
    if audio_file != '.DS_Store': # Fucking OSX
        print(audio_file)

        # Open audio file
        data, sr = sf.read(os.path.join(data_folder_path,audio_file))

        # Perform data analysis
        result = process(data, sr)

        # Write results to file
        result_file_name = os.path.splitext(audio_file)[0] + result_file_extension
        result_file_path = os.path.join(result_folder_path, result_file_name)

        # Adapted from metrics.evaluation_metrics.write_output_format_file
        _fid = open(result_file_path, 'w')
        for _frame_ind in result.keys():
            _value = result[_frame_ind]
        # _ _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
        _fid.close()

print('-------------- PROCESSING FINISHED --------------')

    # # Find associated metadata file
    # file_name = os.path.splitext(audio_file)[0]
    # metadata_file = os.path.join(metadata_folder_path,file_name+metadata_extension)
    # assert os.path.isfile(metadata_file)
    #
    # # Parse metadata file
    # groundtruth = Groundtruth()
    # with open(metadata_file, 'rb') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     for row in reader:
    #         groundtruth.add(row)
