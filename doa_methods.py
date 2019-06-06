import numpy as np
import sys
pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
import parametric_spatial_audio_processing as psa
import matplotlib.pyplot as plt
import scipy.stats

from utils import *
from file_utils import build_result_dict_from_metadata_array, build_metadata_result_array_from_event_dict
from seld_dcase2019_master.metrics.evaluation_metrics import distance_between_spherical_coordinates_rad



def preprocess(data, sr, params):
    """
    Assert first order ambisonics and dimensionality order.
    Compute Stft.
    :param data: np.array (num_frames, num_channels)
    :param sr:  sampling rate
    :param params: params dict
    :return: psa.Stft instance
    """
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

    if params['plot']:
        psa.plot_magnitude_spectrogram(X)

    return X



def estimate_doa(data, sr, params):
    """
    Given an input audio, get the most significant tf bins per frame
    :param data: np.array (num_frames, num_channels)
    :param sr:  sampling rate
    :param params: params dict
    :return: an array in the form :
        [frame, [class_id, azi, ele],[class_id, azi, ele]... ]
        without repeated frame instances, quantized at hop_size,
        containing all valid tf bins doas.
    """

    ### Preprocess data
    X = preprocess(data, sr, params)
    N = X.get_num_time_bins()
    K = X.get_num_frequency_bins()
    r = params['r']

    ### Diffuseness mask
    doa = psa.compute_DOA(X)
    directivity = X.compute_ita_re(r=r)
    directivity_mask = directivity.compute_mask(th=params['directivity_th'])


    ### Energy density mask
    e = psa.compute_energy_density(X)
    block_size = params['energy_density_local_th_size']
    tl = e.compute_threshold_local(block_size=block_size)
    e_mask = e.compute_mask(tl)


    ### DOA Variance mask (computed on azimuth variance)
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

    # TODO: optimize the for loop
    std = np.zeros((K, N))
    doa0_k_array = []
    for r in range(-r_n,r_n+1):
        doa0_k_array.append(np.roll(doa.data[0,:,:],r))
    doa0_k = np.stack(doa0_k_array, axis=0)

    for k in range(r_k, K - r_k):
        std[k, :] = scipy.stats.circstd(doa0_k[:, k - r_k:k + r_k + 1, :], high=np.pi, low=-np.pi, axis=(0, 1))

    # not optimized version...
    # for k in range(r_k, K-r_k):
    #     for n in range(r_n, N-r_n):
    #         # azi
    #         std[k, n] = scipy.stats.circstd(doa.data[0, k-r_k:k+r_k+1, n-r_n:n+r_n+1], high=np.pi, low=-np.pi)
    #         # ele
    #         # std[k, n] = np.std(doa.data[1, k-r_k:k+r_k+1, n-r_n:n+r_n+1])

    # Edges: largest value
    std_max = np.max(std)
    std[0:r_k, :] = std_max
    std[K-r_k:K, :] = std_max
    std[:, 0:r_n] = std_max
    std[:, N - r_n:N] = std_max
    # Scale values to min/max
    std_scaled = std / std_max
    # Invert values
    std_scaled_inv = 1 - std_scaled

    # Compute mask
    doa_std = psa.Stft(doa.t, doa.f, std_scaled_inv, doa.sample_rate)
    doa_std_mask = doa_std.compute_mask(th=params['doa_std_th'])
    mask_all = doa_std_mask.apply_mask(directivity_mask).apply_mask(e_mask)
    doa_th = doa.apply_mask(mask_all)


    ## Median average
    median_averaged_doa = np.empty(doa.data.shape)
    median_averaged_doa.fill(np.nan)
    vicinity_size = (2*r_k-1) + (2*r_n-1)
    doa_median_average_nan_th = params['doa_median_average_nan_th']

    vicinity_radius = params['median_filter_vicinity_radius']
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

    # TODO: optimize the for loop
    for k in range(r_k, K - r_k):
        for n in range(r_n, N - r_n):
            azis = discard_nans(doa_th.data[0, k - r_k:k + r_k + 1, n - r_n:n + r_n + 1].flatten())
            if azis.size > vicinity_size * doa_median_average_nan_th:
                median_averaged_doa[0, k, n] = circmedian(azis, 'rad')
            eles = discard_nans(doa_th.data[1, k - r_k:k + r_k + 1, n - r_n:n + r_n + 1].flatten())
            if eles.size > vicinity_size * doa_median_average_nan_th:
                median_averaged_doa[1, k, n] = np.median(eles)
    doa_th_median = psa.Stft(doa.t, doa.f, median_averaged_doa, doa.sample_rate)


    ## Plot stuff
    if params['plot']:
        psa.plot_doa(doa, title='doa')
        psa.plot_doa(doa.apply_mask(e_mask), title='e mask')
        psa.plot_doa(doa.apply_mask(directivity_mask), title='directivity mask')
        psa.plot_doa(doa.apply_mask(doa_std_mask), title='doa std mask')
        psa.plot_doa(doa_th, title='doa mask all')
        psa.plot_doa(doa_th_median, title='doa circmedian')
        plt.show()


    ## Fold values into a vector

    # Get a list of bins with the position estimation according to the selected doa_method
    # TODO: OPTIMIZE
    active_windows = []
    position = []
    for n in range(N):
        azi = discard_nans(doa_th_median.data[0, :, n])
        ele = discard_nans(doa_th_median.data[1, :, n])
        if np.size(azi) < params['num_min_valid_bins']:
            # Empty! not enough suitable doa values in this analysis window
            pass
        else:
            active_windows.append(n)
            position.append([rad2deg(azi), rad2deg(ele)])

    # result = [bin, class_id, azi, ele] with likely repeated bin instances
    result = []
    label = params['default_class_id']
    for window_idx, window in enumerate(active_windows):
        num_bins = np.shape(position[window_idx])[1]
        for b in range(num_bins):
            azi = position[window_idx][0][b]
            ele = position[window_idx][1][b]
            result.append([window, label, azi, ele])


    # Perform the window transformation by averaging within frame
    ## TODO: assert our bins are smaller than required ones

    current_window_hop = (params['window_size'] - params['window_overlap']) / float(sr)
    window_factor = params['required_window_hop'] / current_window_hop

    # Since frames are ordered (at least they should), we can optimise that a little bit
    last_frame = -1
    # result_quantized = [frame, [class_id, azi, ele],[class_id, azi, ele]... ] without repeated bin instances
    result_quantized = []
    for row in result:
        frame = row[0]
        new_frame = int(np.floor(frame / window_factor))
        if new_frame == last_frame:
            result_quantized[-1].append([row[1], row[2], row[3]])
        else:
            result_quantized.append([new_frame, [row[1], row[2], row[3]]])
        last_frame = new_frame

    return result_quantized




# Assumes overlapping, compute (1,2)-Kmeans on each segment
def group_events(result_quantized, params):
    """
    Segmentate an array of doas into events
    :param result_quantized: an array containing frames and doas
            in the form [frame, [class_id, azi, ele],[class_id, azi, ele]... ]
            without repeated frame instances, with ordered frames
    :param params: params dict
    :return: metadata_result_array, result_dict
            metadata_result_array: array with one event per row, in the form
            [sound_event_recording,start_time,end_time,ele,azi,dist]
            result_dict: dict with one frame per key, in the form:
            {frame: [class_id, azi, ele] or [[class_id1, azi1, ele1], [class_id2, azi2, ele2]]}
    """

    ## Generate result_averaged_dict: grouping doas per frame into 1 or 2 clusters
    ## result_averaged_dict = {frame: [label, azi, ele] or [[label, azi1, ele1],label, azi2, ele2]]}
    result_averaged_dict = {}
    frames = []
    for row in result_quantized:
        frames.append(row[0])

    std_azis = []
    std_eles = []
    std_all = []
    std_th = params['min_std_overlapping']
    label = params['default_class_id']

    for r_idx, row in enumerate(result_quantized):
        # Get all doas
        frame = row[0]
        azis = []
        eles = []
        for v in row[1:]:
            azis.append(v[1])
            eles.append(v[2])
        # Compute std of doas
        std_azis.append(scipy.stats.circstd(azis, high=180, low=-180))
        std_eles.append(np.std(eles))
        std_all.append(std_azis[-1]/2 + std_eles[-1])

        # If big std, we assume 2-overlap
        if std_all[-1] >= std_th:
            # 2 clusters:
            x = deg2rad(np.asarray([azis, eles]).T)
            try:
                kmeans2 = HybridKMeans(n_init=params['num_init_kmeans']).fit(x)
            except RuntimeWarning:
                # All points in x are equal...
                result_averaged_dict[frame] = [label, rad2deg(x[0,0]), rad2deg(x[0,1])]
                continue
            # Keep the centroids of this frame
            result_averaged_dict[frame] = []
            for c in kmeans2.cluster_centers_:
                azi = rad2deg(c[0])
                ele = rad2deg(c[1])
                result_averaged_dict[frame].append([label, azi, ele])
        else:
            # 1 cluster: directly compute the median and keep it
            azi = circmedian(np.asarray(azis), unit='deg')
            ele = np.median(eles)
            result_averaged_dict[frame] = [label, azi, ele]

    if params['plot']:
        plt.figure()
        plt.suptitle('kmeans stds')
        plt.scatter(frames,std_all,label='all')
        plt.axhline(y=std_th)
        plt.legend()
        plt.grid()
        plt.show()


    ## Group doas by distance and time proximity

    # Generate event_dict = { event_id: [ [label, azi_frame, ele_frame] ...}
    # each individual event is a key, and values is a list of [frame, azi, ele]

    d_th = params['max_angular_distance_within_event']
    frame_th = params['max_frame_distance_within_event']
    event_idx = 0
    event_dict = {}

    # Ensure ascending order
    frames = result_averaged_dict.keys()
    frames.sort()
    # TODO: write in a more modular way
    for frame in frames:
        value = result_averaged_dict[frame]
        if len(value) == 3:
            # One source
            azi = value[1]
            ele = value[2]
            if not bool(event_dict):
                # Empty: append
                event_dict[event_idx] = [[frame, azi, ele]]
                event_idx += 1
            else:
                # Compute distance with all previous frames
                new_event = True # default
                for idx in range(event_idx):
                    # Compute distance with median of all previous
                    azis = np.asarray(event_dict[idx])[:, 1]
                    eles = np.asarray(event_dict[idx])[:, 2]
                    median_azi = circmedian(azis, unit='deg')
                    median_ele = np.median(eles)
                    d = distance_between_spherical_coordinates_rad(deg2rad(median_azi),
                                                                   deg2rad(median_ele),
                                                                   deg2rad(azi),
                                                                   deg2rad(ele))

                    last_frame, last_azi, last_ele = event_dict[idx][-1]
                    if d < d_th and abs(frame - last_frame) < frame_th:
                        # Same event
                        new_event = False
                        event_dict[idx].append([frame, azi, ele])
                        break
                if new_event:
                    event_dict[event_idx] = [[frame, azi, ele]]
                    event_idx += 1

        elif len(value) == 2:
            # Two sources
            for v in value:
                azi = v[1]
                ele = v[2]
                if not bool(event_dict):
                    # Empty: append
                    event_dict[event_idx] = [[frame, azi, ele]]
                    event_idx += 1
                    # print(event_dict)
                else:
                    # Compute distance with previous frame
                    new_event = True
                    for idx in range(event_idx):
                        # Compute distance with median of all previous frames
                        azis = np.asarray(event_dict[idx])[:, 1]
                        eles = np.asarray(event_dict[idx])[:, 2]
                        median_azi = circmedian(azis, unit='deg')
                        median_ele = np.median(eles)
                        d = distance_between_spherical_coordinates_rad(deg2rad(median_azi),
                                                                       deg2rad(median_ele),
                                                                       deg2rad(azi),
                                                                       deg2rad(ele))

                        last_frame, last_azi, last_ele = event_dict[idx][-1]
                        if d < d_th and abs(frame - last_frame) < frame_th:
                            # Same event
                            new_event = False
                            event_dict[idx].append([frame, azi, ele])
                            break
                    if new_event:
                        event_dict[event_idx] = [[frame, azi, ele]]
                        event_idx += 1


    ## Explicitly avoid overlapping > 2

    # Generate event_dict_no_overlap: pop doas (in ascending order) if more than 2 overlapping events
    # TODO: more sophisticated algorithm based on event confidence or similar

    # Get max frame (it might be over 3000)
    max_frame = 0
    for event_idx, event_values in event_dict.iteritems():
        end_frame = event_values[-1][0]
        if end_frame >= max_frame:
            max_frame = end_frame

    # Compute the number of events per frame
    events_per_frame = []
    for i in range(max_frame+1):
        events_per_frame.append([])
    for event_idx, event_values in event_dict.iteritems():
        start_frame = event_values[0][0]
        end_frame = event_values[-1][0]
        for frame in range(start_frame, end_frame + 1):
            events_per_frame[frame].append(event_idx)

    # Pop exceeding events
    for i, e in enumerate(events_per_frame):
        while len(e) > 2:
            e.pop()

    # Build event_dict_no_overlap from events_per_frame
    event_dict_no_overlap = {}
    for event_idx, event_values in event_dict.iteritems():
        event_dict_no_overlap[event_idx] = []
        for e in event_values:
            frame = e[0]
            if event_idx in events_per_frame[frame]:
                event_dict_no_overlap[event_idx].append(e)



    ## Filter events to eliminate the spureous ones
    event_dict_filtered = {}
    filtered_event_idx = 0
    min_frames = params['min_num_frames_per_event']
    for frame, v in event_dict_no_overlap.iteritems():
        if len(v) >= min_frames:
            event_dict_filtered[filtered_event_idx] = event_dict_no_overlap[frame]
            filtered_event_idx += 1


    ## Build metadata result array
    offset = params['frame_offset']
    if np.size(offset) == 1:
        pre_offset = post_offset = offset
    elif np.size(offset) == 2:
        # Rectangle! [k, n]
        pre_offset = offset[0]
        post_offset= offset[1]
    else:
        Warning.warn()

    hop_size = params['required_window_hop']  # s
    metadata_result_array = build_metadata_result_array_from_event_dict(event_dict_filtered,
                                                                        label,
                                                                        hop_size,
                                                                        pre_offset,
                                                                        post_offset)

    ## Build result dictionary
    result_dict = build_result_dict_from_metadata_array(metadata_result_array,
                                                        hop_size)

    return metadata_result_array, result_dict
