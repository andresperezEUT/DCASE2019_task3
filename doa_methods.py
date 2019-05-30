import os.path
import soundfile as sf
import numpy as np
import csv
import sys

from file_utils import build_result_dict_from_metadata_array
from seld_dcase2019_master.metrics.evaluation_metrics import distance_between_spherical_coordinates_rad

pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
import parametric_spatial_audio_processing as psa
import matplotlib.pyplot as plt
import sklearn
import scipy.stats
from utils import *



def preprocess(data, sr, params):
    # Assert first order ambisonics and dimensionality order
    num_frames = np.shape(data)[0]
    num_channels = np.shape(data)[1]
    assert num_channels == 4

    start_frame = 0
    if params['quick_test']:
        end_frame = int(np.ceil(sr * params['quick_test_file_duration']))
    else:
        end_frame = num_frames

    window_size = params['window_size']
    window_overlap = params['window_overlap']
    nfft = params['nfft']

    x = psa.Signal(data[start_frame:end_frame].T, sr, 'acn', 'n3d')
    X = psa.Stft.fromSignal(x,
                            window_size=window_size,
                            window_overlap=window_overlap,
                            nfft=nfft
                            ).limit_bands(params['fmin'], params['fmax'])

    if params['plot']: psa.plot_magnitude_spectrogram(X)

    return X

def fold_doa_into_results_vector(doa_th, sr, params, method='mean'):
    """

    :param doa_th:
    :param sr:
    :param params:
    :param method:
    :return:
        result: an array of type [frame_idx, class_id, azimuth, elevation] in the analysis window hop time,
        possibly with repeated frame_idx values
        result_quantized: an array of type [frame_idx, [[class_id, azimuth, elevation]] in the target window hop time,
        with a row for each different frame_idx, time-quantizing all values given in result
        # result_averaged_dict: a dictionary of the form {frame_idx: [[class_id, azimuth elevation]]}  in the target window hop time,
        # with a key for each different frame_idx. It averages the content of result_quantized for one or two dict values per key.
    """

    print('folding values...')
    N = doa_th.get_num_time_bins()
    K = doa_th.get_num_frequency_bins()

    active_windows = []
    position = []
    position_processed = []

    ## Get a list of bins with the position estimation according to the selected doa_method

    for n in range(N):
        azi = discard_nans(doa_th.data[0, :, n])
        ele = discard_nans(doa_th.data[1, :, n])

        if np.size(azi) < params['num_min_valid_bins']:
            # Empty! not enough suitable doa values
            pass
        else:
            active_windows.append(n)
            position.append([rad2deg(azi), rad2deg(ele)])

    # result = [bin, class_id, azi, ele] with likely repeated bin instances
    result = []
    class_id = params['default_class_id']
    for window_idx, window in enumerate(active_windows):
        num_bins = np.shape(position[window_idx])[1]
        for b in range(num_bins):
            azi = position[window_idx][0][b]
            ele = position[window_idx][1][b]
            result.append([window, class_id, azi, ele])



    ## Perform the window transformation by averaging within new bin
    ## TODO: assert our bins are smaller than required ones

    current_window_hop = (params['window_size'] - params['window_overlap']) / float(sr)
    window_factor = params['required_window_hop'] / current_window_hop

    # Since bins are ordered (at least they should), we can optimise that a little bit
    last_bin = -1
    # result_quantized = [bin, [class_id, azi, ele],[class_id, azi, ele]... ] without repeated bin instances
    result_quantized = []
    for row in result:
        bin = row[0]
        new_bin = int(np.floor(bin / window_factor))
        if new_bin == last_bin:
            result_quantized[-1].append([row[1], row[2], row[3]])
        else:
            result_quantized.append([new_bin, [row[1], row[2], row[3]]])
        last_bin = new_bin



    # result_averaged_dict = {}
    #
    # for row in result_quantized:
    #     bin = row[0]
    #     label = 0  # TODO
    #
    #     azis = np.asarray(row[1:])[:, 1]
    #     eles = np.asarray(row[1:])[:, 2]
    #
    #     if method == 'kmeans':
    #         x = np.asarray([azis, eles]).T
    #         # print (bin)
    #         kmeans1 = KMeans(n_clusters=1, random_state=0).fit(x)
    #         kmeans2 = KMeans(n_clusters=2, random_state=0).fit(x)
    #         rate = kmeans1.inertia_ / kmeans2.inertia_
    #
    #         if rate > params['kmeans_rate_th']:
    #             # 2 clusters:
    #             result_averaged_dict[bin] = []
    #             for c in kmeans2.cluster_centers_:
    #                 print(bin, azi, ele)
    #                 azi = c[0]
    #                 ele = c[1]
    #                 result_averaged_dict[bin].append([label, azi, ele])
    #                 print result_averaged_dict[bin]
    #         else:
    #             # 1 cluster: median
    #             azi = circmedian(azis, unit='deg')
    #             ele = np.median(eles)
    #             result_averaged_dict[bin] = [label, azi, ele]
    #
    #     ## Single source approach
    #     elif method == 'median':
    #         azi = circmedian(azis, unit='deg')
    #         ele = np.median(eles)
    #         result_averaged_dict[bin] = [label, azi, ele]
    #
    #     elif method == 'mean':
    #         azi = scipy.stats.circmean(azis, high=180, low=-180)
    #         ele = np.mean(eles)
    #         result_averaged_dict[bin] = [label, azi, ele]

    return result, result_quantized




##### TODO!!!!
## ADAPT TO OVERLAPPING CASE
## COMPUTE CONTINUITY ON THE WHOLE BAND WITHOUT PREVIOUS PROCESSING...

def find_regions(result_averaged_dict, th=30):

    starts = []
    ends = []
    frames = result_averaged_dict.keys()
    # Ensure ascending order
    frames.sort()

    print(frames)
    # Add first start
    # starts.append(frames[0])

    for i, f in enumerate(frames[:-1]):
        print('---')
        print (f, f - frames[i-1])
        if abs(f - frames[i-1]) > th :
            # start
            starts.append(f)
        print (f,frames[i + 1] - f)
        if frames[i + 1] - f > th :
            # end
            ends.append(f)

        print(starts, ends)

    # Add last end
    ends.append(frames[-1])
    print('++++++++')
    print(starts, ends)

    assert len(starts) == len(ends)

    edges = []
    for s, e in zip(starts, ends):
        edges.append([s, e])

    return edges


# Assumes no overlapping: compute mean of each region (source activity)
def group_sources(result_averaged_dict):

    hop_size = 0.02 # s

    sources_array = []
    regions = find_regions(result_averaged_dict)
    metadata_result_array = []

    # find mean of each region
    for r in regions:
        start = r[0]
        end = r[1]

        azis = []
        eles = []
        for frame, value in result_averaged_dict.iteritems():
            if frame >= start and frame <= end:
                # print (frame, value)
                azis.append(value[1])
                eles.append(value[2])
                # eles.append(value[2])
        print(azis)
        # print(eles)
        # azis = result_averaged_dict.

        mean_azi = scipy.stats.circmean(azis, high=180, low=-180)
        std_azi = scipy.stats.circstd(azis, high=180, low=-180)
        mean_ele = np.mean(eles)
        std_ele = np.std(eles)

        metadata_result_array.append([ None, start*hop_size, end*hop_size, mean_ele, mean_azi, None ])

    return metadata_result_array



def find_regions_q(result_quantized, th=30):

    starts = []
    ends = []
    frames = []
    for row in result_quantized:
        frames.append(row[0])
    # Ensure ascending order
    frames.sort()

    # print(frames)
    # Add first start
    # starts.append(frames[0])

    for i, f in enumerate(frames[:-1]):
        # print('---')
        # print (f, f - frames[i-1])
        if abs(f - frames[i-1]) > th :
            # start
            starts.append(f)
        # print (f,frames[i + 1] - f)
        if frames[i + 1] - f > th :
            # end
            ends.append(f)

        # print(starts, ends)

    # Add last end
    ends.append(frames[-1])
    # print('++++++++')
    # print(starts, ends)

    assert len(starts) == len(ends)

    edges = []
    for s, e in zip(starts, ends):
        edges.append([s, e])

    return edges


# Assumes 2-overlapping, but only in the whole range of activity continuity
# Build result_averaged_dict by median-averaging the whole activity range
def group_sources_q(result_quantized, params):

    hop_size = params['required_window_hop'] # s
    file_duration = params['file_duration'] # s
    regions = find_regions_q(result_quantized)
    metadata_result_array = []
    result_averaged_dict = {}

    azis = []
    eles = []

    # find mean of each region
    for r_idx, r in enumerate(regions):
        # print('region '+str(r_idx))
        start = r[0]
        end = r[1]
        # print('start ',start)
        # print('end ',end)

        # Get azis and eles for each time region
        azis.append([])
        eles.append([])
        for row in result_quantized:
            frame = row[0]
            if frame >= start and frame <= end:
                # print(frame)
                for v in row[1:]:
                    # print(v[1:])
                    azis[r_idx].append(v[1])
                    eles[r_idx].append(v[2])

        # Kmeans expects values as degrees
        x = deg2rad(np.asarray([azis[r_idx], eles[r_idx]]).T)
        kmeans1 = HybridKmeans_implementation(k=1, max_iter=100).run(x)
        kmeans2 = HybridKmeans_implementation(k=2, max_iter=100).run(x)
        rate = kmeans1.inertia_ / kmeans2.inertia_
        # print(rate)

        cluster_centers = kmeans2.cluster_centers_ if rate > params['kmeans_rate_th'] else kmeans1.cluster_centers_

        # Output values are expected in degrees
        for c in cluster_centers:
            c = rad2deg(c)
            metadata_result_array.append([None, start * hop_size, end * hop_size, c[1], c[0], None])

        # Build averaged dict
        # Add all missing time windows...
        num_frames = file_duration/hop_size
        for frame in range(int(num_frames)):
        # for row in result_quantized:
        #     frame = row[0]
            if frame >= start and frame <= end:
                    result_averaged_dict[frame] = [0, c[0], c[1]]

        # for frame, value in result_quantized.iteritems():
        #     if frame >= start and frame <= end:
        #         # print (frame, value)
        #         azis.append(value[1])
        #         eles.append(value[2])
        #         # eles.append(value[2])
        # print(azis)
        # print(eles)
        # azis = result_averaged_dict.

        # mean_azi = scipy.stats.circmean(azis, high=180, low=-180)
        # std_azi = scipy.stats.circstd(azis, high=180, low=-180)
        # mean_ele = np.mean(eles)
        # std_ele = np.std(eles)
        #
        # metadata_result_array.append([ None, start*hop_size, end*hop_size, mean_ele, mean_azi, None ])

    return metadata_result_array, result_averaged_dict



# Assumes overlapping, compute (1,2)-Kmeans on each segment
def group_sources_q_overlap(result_quantized, params):
    print('grouping sources...')
    result_averaged_dict = {}

    # TODO: MORE PYTHONIC
    frames = []
    for row in result_quantized:
        frames.append(row[0])
    rates = []
    inertia1 = []
    inertia2 = []

    std_azis = []
    std_eles = []
    std_all = []

    std_th = params['min_std_overlapping']
    label = 0

    for r_idx, row in enumerate(result_quantized):
        frame = row[0]
        azis = []
        eles = []

        for v in row[1:]:
            # print(v[1:])
            azis.append(v[1])
            eles.append(v[2])

        # x = deg2rad(np.asarray([azis, eles]).T)
        # kmeans1 = HybridKmeans_implementation(k=1, max_iter=100).run(x)
        # kmeans2 = HybridKmeans_implementation(k=2, max_iter=100).run(x)
        # rate = kmeans1.inertia_ / kmeans2.inertia_

        std_azis.append(scipy.stats.circstd(azis, high=180, low=-180))
        std_eles.append(np.std(eles))
        std_all.append(std_azis[-1]/2 + std_eles[-1])

        if std_all[-1] >= std_th:
            # 2 clusters:
            x = deg2rad(np.asarray([azis, eles]).T)
            kmeans2 = HybridKmeans_implementation(k=2, max_iter=100).run(x)
            result_averaged_dict[frame] = []
            for c in kmeans2.cluster_centers_:
                # print(frame, azi, ele)
                azi = rad2deg(c[0])
                ele = rad2deg(c[1])
                result_averaged_dict[frame].append([label, azi, ele])
                # print result_averaged_dict[frame]
        else:
            # 1 cluster: median
            azi = circmedian(np.asarray(azis), unit='deg')
            ele = np.median(eles)
            result_averaged_dict[frame] = [label, azi, ele]

    plt.figure()
    plt.suptitle('kmeans stds')
    # plt.scatter(frames,std_azis,label='azis')
    # plt.scatter(frames,std_eles,label='eles')
    plt.scatter(frames,std_all,label='all')
    plt.axhline(y=std_th)
    # plt.scatter(frames,np.asarray(std_azis)/2*np.asarray(std_eles),label='mult')
    plt.legend()
    plt.grid()
    plt.show()

    metadata_result_array = None

    # Group stuff

    # event_dict: each individual event is a key, and values is a list of [frame, azi, ele]

    d_th = 10 # for example
    frame_th = 20 # for example
    event_idx = 0
    event_dict = {}

    # Ensure ascending order
    frames = result_averaged_dict.keys()
    frames.sort()

    # for frame, value in result_averaged_dict.iteritems():
    for frame in frames:
        value = result_averaged_dict[frame]
        # One source
        if len(value) == 3:
            azi = value[1]
            ele = value[2]
            # print('++++++')
            # print(frame, azi, ele)
          # Compute distance with previous frame
            if not bool(event_dict): # Empty: append
                event_dict[event_idx] = [[frame, azi, ele]]
                # event_idx += 1
                # print(event_dict)
            else:
                # Get all last values
                new_event = True
                for idx in range(event_idx):
                    # print('check event '+str(idx))
                    last_frame, last_azi, last_ele = event_dict[idx][-1]
                    d = distance_between_spherical_coordinates_rad(deg2rad(last_azi),
                                                                   deg2rad(last_ele),
                                                                   deg2rad(azi),
                                                                   deg2rad(ele))
                    # print('distances', d, azi, ele, last_azi, last_ele)
                    # print (d, abs(frame - last_frame))
                    if d < d_th and abs(frame - last_frame) < frame_th:
                    # if d < d_th:
                        # Same event
                        new_event = False
                        event_dict[idx].append([frame, azi, ele])
                        # print(event_dict)
                        break
                if new_event:
                    # print('new event')
                    event_dict[event_idx] = [[frame, azi, ele]]
                    event_idx += 1
                    # print(event_dict)

        elif len(value) == 2:
            # Two sources
            for v in value:
                azi = v[1]
                ele = v[2]
                # print('++++++')
                # print(frame, azi, ele)
                # Compute distance with previous frame
                if not bool(event_dict):  # Empty: append
                    event_dict[event_idx] = [[frame, azi, ele]]
                    event_idx += 1
                    # print(event_dict)
                else:
                    # Get all last values
                    new_event = True
                    for idx in range(event_idx):
                        # print('check event ' + str(idx))
                        last_frame, last_azi, last_ele = event_dict[idx][-1]
                        d = distance_between_spherical_coordinates_rad(deg2rad(last_azi),
                                                                       deg2rad(last_ele),
                                                                       deg2rad(azi),
                                                                       deg2rad(ele))
                        # print('distances', d, azi, ele, last_azi, last_ele)
                        # print (d, abs(frame - last_frame))
                        if d < d_th and abs(frame - last_frame) < frame_th:
                        # if d < d_th:
                            # Same event
                            new_event = False
                            event_dict[idx].append([frame, azi, ele])
                            # print(event_dict)
                            break
                    if new_event:
                        # print('new event')
                        event_dict[event_idx] = [[frame, azi, ele]]
                        event_idx += 1
                        # print(event_dict)


    # Build metadata result array
    metadata_result_array = []
    class_id = params['default_class_id']
    hop_size = params['required_window_hop']  # s
    for event_idx, event_values in event_dict.iteritems():
        start_frame = event_values[0][0]
        end_frame = event_values[-1][0]

        azis = np.asarray(event_values)[:,1]
        azi = circmedian(np.asarray(azis), unit='deg')
        eles = np.asarray(event_values)[:,2]
        ele = np.median(eles)

        metadata_result_array.append([class_id, start_frame * hop_size, end_frame * hop_size, ele, azi, None])


    # num_frames = file_duration / hop_size
    # for frame in range(int(num_frames)):
    #     # for row in result_quantized:
    #     #     frame = row[0]
    #     if frame >= start and frame <= end:
    #         result_averaged_dict[frame] = [0, c[0], c[1]]


    # TODO
    # # Re-build averaged dict
    # # Add all missing time windows and average values
    result_dict = build_result_dict_from_metadata_array(metadata_result_array, hop_size)


    # return metadata_result_array, result_averaged_dict
    return metadata_result_array, result_dict


### TODO TODO TODO discriminate here also from angle
def find_regions_position(result_averaged_dict, frame_th=1, dist_th=10):

    starts = []
    ends = []
    frames = result_averaged_dict.keys()
    # Ensure ascending order
    frames.sort()

    print(frames)
    # Add first start
    # starts.append(frames[0])

    for i, f in enumerate(frames[:-1]):
        print('---')
        print (f, f - frames[i-1])
        if abs(f - frames[i-1]) > frame_th :
            # start
            starts.append(f)
        print (f,frames[i + 1] - f)
        if frames[i + 1] - f > frame_th :
            # end
            ends.append(f)

        print(starts, ends)

    # Add last end
    ends.append(frames[-1])
    # print('++++++++')
    # print(starts, ends)

    assert len(starts) == len(ends)

    edges = []
    for s, e in zip(starts, ends):
        edges.append([s, e])

    return edges




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# METHODS
#
# def doa_method_variance2(data, sr, params):
#
#     X = preprocess(data, sr, params)
#     N = X.get_num_time_bins()
#     K = X.get_num_frequency_bins()
#     r = params['r']
#
#     # Diffuseness threshold
#     doa = psa.compute_DOA(X)
#     directivity = X.compute_ksi_re(r=r)
#     directivity_mask = directivity.compute_mask(th=params['directivity_th'])
#
#     if params['plot']: psa.plot_doa(doa, title=str(r))
#     if params['plot']: plt.show()
#
#     # Doa Variance
#     doa_std_vicinity_radius = params['doa_std_vicinity_radius']
#     if np.size(doa_std_vicinity_radius) == 1:
#         # Square!
#         r_k = doa_std_vicinity_radius
#         r_n = doa_std_vicinity_radius
#     elif np.size(doa_std_vicinity_radius) == 2:
#         # Rectangle! [k, n]
#         r_k = doa_std_vicinity_radius[0]
#         r_n = doa_std_vicinity_radius[1]
#     else:
#         Warning.warn()
#
#     std = np.zeros((2, K, N))
#
#     for k in range(r_k, K-r_k):
#         for n in range(r_n, N-r_n):
#
#             # range_k = range(k-r_k, k+r_k+1)
#             # range_n = range(n-r_n, n+r_n+1)
#             std[0, k, n] = scipy.stats.circstd(doa.data[0, k-r_k:k+r_k+1, n-r_n:n+r_n+1], high=np.pi, low=-np.pi)
#             std[1, k, n] = np.std(doa.data[1, k-r_k:k+r_k+1, n-r_n:n+r_n+1])
#
#     # Edges: large value
#     max_v = [np.max(std[0]), np.max(std[1])]
#     for m in range(2):
#         for k in range(0, r_k):
#             # std[m, k, :] = std[m, k+doa_std_vicinity_radius, :]
#             std[m, k, :] = max_v[m]
#         for k in range(K-r_k, K):
#             # std[m, k, :] = std[m, K-doa_std_vicinity_radius-1, :]
#             std[m, k, :] = max_v[m]
#         for n in range(0, r_n):
#             # std[m, :, n] = std[m, :, n + doa_std_vicinity_radius]
#             std[m, :, n] = max_v[m]
#         for n in range(N - r_n, N):
#             # std[m, :, n] = std[m, :, N - doa_std_vicinity_radius - 1,]
#             std[m, :, n] = max_v[m]
#
#     # if params['plot']:
#     #     plt.figure(figsize=plt.figaspect(1 / 2.))
#     #     plt.subplot(211)
#     #     plt.pcolormesh(std[0])
#     #     plt.colorbar()
#     #     plt.subplot(212)
#     #     plt.pcolormesh(std[1])
#     #     plt.colorbar()
#     #     plt.show()
#
#     # Scale values to min/max
#     std_azi_max = np.max(std[0])
#     std_ele_max = np.max(std[1])
#
#     std_scaled = np.zeros((2, K, N))
#     std_scaled[0] = std[0] / std_azi_max
#     std_scaled[1] = std[1] / std_ele_max
#
#     # Invert values
#     std_scaled_inv = 1 - std_scaled
#
#     if params['plot']:
#         plt.figure(figsize=plt.figaspect(1 / 2.))
#         plt.subplot(211)
#         plt.pcolormesh(std_scaled_inv[0])
#         plt.colorbar()
#         plt.subplot(212)
#         plt.pcolormesh(std_scaled_inv[1])
#         plt.colorbar()
#         plt.show()
#
#     doa_std_azi = psa.Stft(doa.t, doa.f, std_scaled_inv[0], doa.sample_rate)
#     doa_std_mask_azi = doa_std_azi.compute_mask(th=params['doa_std_th'])
#     # doa_std_ele = psa.Stft(doa.t, doa.f, std_scaled_inv[1], doa.sample_rate)
#     # doa_std_mask_ele = doa_std_ele.compute_mask(th=params['doa_std_th'])
#     # doa_std_mask = doa_std_mask_ele.apply_mask(doa_std_mask_azi)
#     doa_std_mask = doa_std_mask_azi
#
#     mask2 = doa_std_mask.apply_mask(directivity_mask)
#     doa_th = doa.apply_mask(mask2)
#
#     if params['plot']:
#         psa.plot_doa(doa, title='doa')
#
#         psa.plot_mask(directivity_mask, title='directivity mask')
#         psa.plot_doa(doa.apply_mask(directivity_mask), title='directivity mask')
#
#         # psa.plot_mask(doa_std_mask_azi, title='doa std mask_azi')
#         # psa.plot_doa(doa.apply_mask(doa_std_mask_azi), title='doa std mask_azi')
#
#         # psa.plot_mask(doa_std_mask_ele, title='doa std mask_ele')
#         # psa.plot_doa(doa.apply_mask(doa_std_mask_ele), title='doa std mask_ele')
#
#         psa.plot_mask(doa_std_mask, title='doa std mask')
#         psa.plot_doa(doa.apply_mask(doa_std_mask), title='doa std mask')
#
#         psa.plot_mask(mask2, title='mask 2')
#         psa.plot_doa(doa_th)
#         plt.show()
#
#     return fold_doa_into_results_vector(doa_th, sr, params, method='kmeans')
#

def doa_method_variance(data, sr, params):

    X = preprocess(data, sr, params)
    N = X.get_num_time_bins()
    K = X.get_num_frequency_bins()
    r = params['r']


    # if params['plot']: plt.pcolormesh(e_masked.data[0])

    print('compute doa, directivity...')
    # Diffuseness threshold
    doa = psa.compute_DOA(X)
    directivity = X.compute_ksi_re(r=r)
    directivity_mask = directivity.compute_mask(th=params['directivity_th'])

    if params['plot']: psa.plot_doa(doa, title=str(r))
    if params['plot']: plt.show()

    print('compute energy density')
    e = psa.compute_energy_density(X)
    block_size = 51
    # for mode in ['gaussian', 'mean', 'median']:
    #     tl = e.compute_threshold_local(block_size=block_size, mode=mode)
    #     e_mask = e.compute_mask(tl)
    #     if params['plot']: psa.plot_magnitude_spectrogram(e, title='e')
    #     if params['plot']: psa.plot_mask(e_mask, title='e_mask'+mode)
    #     plt.show()
    #     doa_e_masked = doa.apply_mask(e_mask)
    #     if params['plot']:  psa.plot_doa(doa_e_masked, title='e mask'+mode)
    #     plt.show()
    tl = e.compute_threshold_local(block_size=block_size)
    e_mask = e.compute_mask(tl)
    if params['plot']: psa.plot_magnitude_spectrogram(e, title='e')
    if params['plot']: psa.plot_mask(e_mask, title='e_mask')
    plt.show()
    doa_e_masked = doa.apply_mask(e_mask)
    if params['plot']:  psa.plot_doa(doa_e_masked, title='e mask')
    plt.show()

    print('compute doa std...')
    # Doa Variance
    # Aximuth std depends on the location!! so let's just use elevation stf
    vicinity_radius = params['doa_std_vicinity_radius']
    if np.size(vicinity_radius) == 1:
        # Square!
        r_k = vicinity_radius
        r_n = vicinity_radius
    elif np.size(vicinity_radius) == 2:
        # Rectangle! [k, n]
        r_k = vicinity_radius[0]
        r_n = vicinity_radius[1]
    else:
        Warning.warn()

    std = np.zeros((K, N))

    for k in range(r_k, K-r_k):
        for n in range(r_n, N-r_n):
            # azi
            std[k, n] = scipy.stats.circstd(doa.data[0, k-r_k:k+r_k+1, n-r_n:n+r_n+1], high=np.pi, low=-np.pi)
            # ele
            # std[k, n] = np.std(doa.data[1, k-r_k:k+r_k+1, n-r_n:n+r_n+1])

    # Edges: large value
    std_max = np.max(std)
    # for k in range(0, r_k):
    #     # std[m, k, :] = std[m, k+doa_std_vicinity_radius, :]
    #     std[k, :] = std_max
    # for k in range(K-r_k, K):
    #     # std[m, k, :] = std[m, K-doa_std_vicinity_radius-1, :]
    #     std[ k, :] = std_max
    # for n in range(0, r_n):
    #     # std[m, :, n] = std[m, :, n + doa_std_vicinity_radius]
    #     std[:, n] = std_max
    # for n in range(N - r_n, N):
    #     # std[m, :, n] = std[m, :, N - doa_std_vicinity_radius - 1,]
    #     std[:, n] = std_max
    std[0:r_k, :] = std_max
    std[K-r_k:K, :] = std_max
    std[:, 0:r_n] = std_max
    std[:, N - r_n:N] = std_max


    # Scale values to min/max
    std_scaled = std / std_max

    # Invert values
    std_scaled_inv = 1 - std_scaled

    if params['plot']:
        plt.pcolormesh(std_scaled_inv)
        plt.colorbar()
        plt.show()


    doa_std = psa.Stft(doa.t, doa.f, std_scaled_inv, doa.sample_rate)
    doa_std_mask = doa_std.compute_mask(th=params['doa_std_th'])

    mask_all = doa_std_mask.apply_mask(directivity_mask).apply_mask(e_mask)
    doa_th = doa.apply_mask(mask_all)

    print('compute doa median average...')

    # MEdian averAGE
    median_averaged_doa = np.empty(doa.data.shape)
    median_averaged_doa.fill(np.nan)
    vicinity_size = (2*r_k-1) + (2*r_n-1)
    doa_median_average_nan_th = params['doa_median_average_nan_th']
    for k in range(r_k, K - r_k):
        for n in range(r_n, N - r_n):

            azis = discard_nans(doa_th.data[0, k - r_k:k + r_k + 1, n - r_n:n + r_n + 1].flatten())
            if azis.size > vicinity_size * doa_median_average_nan_th:
                median_averaged_doa[0, k, n] = circmedian(azis, 'rad')
            # else:
            #     median_averaged_doa[0, k, n] = np.nan

            eles = discard_nans(doa_th.data[1, k - r_k:k + r_k + 1, n - r_n:n + r_n + 1].flatten())
            if eles.size > vicinity_size * doa_median_average_nan_th:
                median_averaged_doa[1, k, n] = np.median(eles)
            # else:
            #     median_averaged_doa[1, k, n] = np.nan

            # median_averaged_doa[0, k, n] = circmedian(discard_nans(doa_th.data[0, k - r_k:k + r_k + 1, n - r_n:n + r_n + 1].flatten()), 'rad')
            # median_averaged_doa[1, k, n] = np.median(discard_nans(doa_th.data[1, k - r_k:k + r_k + 1, n - r_n:n + r_n + 1]))
    # # Edges: nan
    # for k in range(0, r_k):
    #     median_averaged_doa[:, k, :] = np.nan
    # for k in range(K - r_k, K):
    #     median_averaged_doa[:, k, :] = np.nan
    # for n in range(0, r_n):
    #     median_averaged_doa[:, :, n] = np.nan
    # for n in range(N - r_n, N):
    #     median_averaged_doa[:, :, n] = np.nan
    # Convert to Stft
    doa_median = psa.Stft(doa.t, doa.f, median_averaged_doa, doa.sample_rate)

    if params['plot']:
        psa.plot_doa(doa, title='doa')

        psa.plot_doa(doa_e_masked, title='e mask')

        # psa.plot_mask(directivity_mask, title='directivity mask')
        psa.plot_doa(doa.apply_mask(directivity_mask), title='directivity mask')

        # psa.plot_mask(doa_std_mask, title='doa std mask')
        psa.plot_doa(doa.apply_mask(doa_std_mask), title='doa std mask')

        # psa.plot_mask(mask2, title='mask 2')
        psa.plot_doa(doa_th, title='doa mask all')

        psa.plot_doa(doa_median, title='doa circmedian')
        plt.show()

    # return fold_doa_into_results_vector(doa_th, sr, params, method='median')
    return fold_doa_into_results_vector(doa_median, sr, params, method='median')


def doa_method_median(data, sr, params):

    X = preprocess(data, sr, params)
    N = X.get_num_time_bins()
    K = X.get_num_frequency_bins()
    r = params['r']

    doa = psa.compute_DOA(X)
    directivity = X.compute_ksi_re(r=r)
    directivity_mask = directivity.compute_mask(th=params['directivity_th'])
    doa_th = doa.apply_mask(directivity_mask)

    return fold_doa_into_results_vector(doa_th, sr, params, method='median')



def doa_method_mean(data, sr, params):

    X = preprocess(data, sr, params)

    # e = psa.compute_energy_density(x)
    # if params['plot']: psa.plot_signal(e, y_scale='log')

    # env = psa.compute_signal_envelope(e, windowsize=1024)
    # plt.plot(e.data[0])
    # plt.plot(env.data[0])
    # plt.hlines(energy_density_th,0,x.get_num_frames())
    # plt.yscale('log')
    # plt.show()

    ################


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
    # active_windows = []
    # onset_samples = []
    # offset_samples = []
    # for n in x.get_num_frames():
    #     if energy_smooth[n] > energy_th:
    #         active_windows.append()

    N = X.get_num_time_bins()
    K = X.get_num_frequency_bins()
    r = params['r']


    doa = psa.compute_DOA(X)
    directivity = X.compute_ksi_re(r=r)
    directivity_mask = directivity.compute_mask(th=params['directivity_th'])
    doa_th = doa.apply_mask(directivity_mask)

    if params['plot']: psa.plot_doa(doa, title=str(r))
    if params['plot']: psa.plot_doa(doa_th, title=str(r))
    if params['plot']: plt.show()

    N = X.get_num_time_bins()
    K = X.get_num_frequency_bins()

    ## Compute position statistics
    # active_windows = []
    # position_mean = []
    # position_std = []
    #
    # for n in range(N):
    #     azi = discard_nans(doa_th.data[0, :, n]).reshape(-1, 1)
    #     ele = discard_nans(doa_th.data[1, :, n]).reshape(-1, 1)
    #
    #     if np.size(azi) < params['num_min_valid_bins']:
    #         # Empty! not enough suitable doa values
    #         pass
    #     else:
    #         active_windows.append(n)
    #
    #         mean_azi = rad2deg(scipy.stats.circmean(azi, high=np.pi, low=-np.pi))
    #         mean_ele = rad2deg(np.mean(ele))
    #         position_mean.append([mean_azi, mean_ele])
    #
    #         std_azi = rad2deg(scipy.stats.circstd(azi, high=np.pi, low=-np.pi))
    #         std_ele = rad2deg(np.std(ele))
    #         position_std.append([std_azi, std_ele])


    ## Find contiguous regions

    # audio_event_min_separation = 100  # bins @256 window size
    # audio_event_min_duration = 50  # bins @256 window size
    # onsets = []
    # offsets = []
    #
    # for n in range(N):
    #     if n in active_windows:
    #         index_of_current_active_bin = active_windows.index(n)
    #
    #         ## Find onsets
    #         if index_of_current_active_bin == 0:
    #             # This is the first active bin, so probably it's an onset
    #             onsets.append(n)
    #         else:
    #             last_active_bin = active_windows[index_of_current_active_bin - 1]
    #             if (n - last_active_bin) > audio_event_min_separation:
    #                 onsets.append(n)
    #
    #         ## Find offsets
    #         if index_of_current_active_bin == len(active_windows) - 1:
    #             # This is the last active bin, so probably it's an offset
    #             offsets.append(n)
    #         else:
    #             next_active_bin = active_windows[index_of_current_active_bin + 1]
    #             if (next_active_bin - n) > audio_event_min_separation:
    #                 offsets.append(n)
    #
    # assert len(onsets) == len(offsets)
    #
    # # Take only the events larger than the minimum duration (remove spureous and transients)
    # postprocessed_onsets = []
    # postprocessed_offsets = []
    # event_lenghts = np.asarray(offsets) - np.asarray(onsets)
    # for event_idx, event_length in enumerate(event_lenghts):
    #     if event_length > audio_event_min_duration:
    #         postprocessed_onsets.append(onsets[event_idx])
    #         postprocessed_offsets.append(offsets[event_idx])
    #
    # assert len(postprocessed_onsets) == len(postprocessed_offsets)

    # if params['plot']:
    #     ## Plot the nice graphs over time
    #     ymin_azi = -200
    #     ymax_azi = 200
    #     ymin_ele = -100
    #     ymax_ele = 100
    #     with plt.style.context(('seaborn-whitegrid')):
    #         plt.figure(figsize=plt.figaspect(1 / 2.))
    #         plt.suptitle('azimuth')
    #         plt.errorbar(active_windows, np.asarray(position_mean)[:, 0], yerr=np.asarray(position_std)[:, 0], fmt='o',
    #                      color='black',
    #                      ecolor='lightgray', elinewidth=1, capsize=0, markersize=1)
    #         plt.ylim(ymin_azi, ymax_azi)
    #         # plt.vlines(postprocessed_onsets, ymin_azi, ymax_azi, linestyles='dashed', colors='k')
    #         # plt.vlines(postprocessed_offsets, ymin_azi, ymax_azi, linestyles='dashed', colors='b')
    #         plt.show()
    #
    #         plt.figure(figsize=plt.figaspect(1 / 2.))
    #         plt.suptitle('elevation')
    #         plt.errorbar(active_windows, np.asarray(position_mean)[:, 1], yerr=np.asarray(position_std)[:, 1], fmt='o',
    #                      color='black',
    #                      ecolor='lightgray', elinewidth=1, capsize=0, markersize=1)
    #         plt.ylim(ymin_ele, ymax_ele)
    #         # plt.vlines(postprocessed_onsets, ymin_azi, ymax_azi, linestyles='dashed', colors='k')
    #         # plt.vlines(postprocessed_offsets, ymin_azi, ymax_azi, linestyles='dashed', colors='b')
    #         plt.show()

    return fold_doa_into_results_vector(doa_th, sr, params, method='mean')
