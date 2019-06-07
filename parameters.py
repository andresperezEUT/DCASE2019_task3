## Obviously inspired by seld_dcase2019_master.parameter.py

def get_params(preset_string=None):

    # ########### default parameters ##############
    params = dict(

        preset_string = '0',

        # PATHS, NAMING CONVENTIONS
        dataset_dir = './data',                             # Base folder containing the foa/mic and metadata folders
        output_result_folder_path = './results_output',     # Folder for saving final results
        output_result_file_extension = '.csv',
        metadata_result_folder_path='./results_metadata',   # Folder for saving intermediate metadata results
        metadata_result_file_extension='.csv',
        before_classification_folder_name='doa',            # Name for after_classification folders
        after_classification_folder_name='classif',

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',      # 'foa' - ambisonic or 'mic' - microphone signals

        # MODEL PARAMETERS
        default_class_id = -1,
        required_window_hop = 0.02, # in seconds
        file_duration = 60,         # in seconds
        ## doa estimation
        window_size = 256,                  # analysis window size
        fmin = 125,                         # doa estimation min frequency
        fmax = 8000,                        # doa estimation max frequency
        r = 4,                              # vicinity radius for averaging (diffuseness, median filter)
        energy_density_local_th_size = 51,  # number of bins used to compute threshold local in the energy density stft
        directivity_th = 0.75,              # minimum value for the directivity mask
        doa_std_vicinity_radius = 3,        # vicinity radius for the std doa mask
        doa_std_th = 0.75,                  # minimum value for the std doa mask
        doa_median_average_nan_th = 0.75,   # percentage of not-nan vicinity tf bins for computing median averaged doa
        num_min_valid_bins = 10,            # minimum number of masked tf bins per analysis window
        ## event grouping
        num_init_kmeans = 10,                   # KMeans run iterations
        frame_offset = [0,0],                   # pre-post frame offset when building metadata_result_array
        max_angular_distance_within_event = 20, # in degrees, maximum value for consider two events as the same
        max_frame_distance_within_event = 20,   # in frames, maximum value for consider two events as the same
        min_num_frames_per_event = 10,          # in frames, minimum number of hop_size frames for an event
        min_std_overlapping = 30,               # minimum angular position std value per frame to consider 2-overlap

        # UTILS
        ## Grountruth class acquisition (dev only)
        gt_time_th = 1.,    # in seconds
        gt_dist_th = 20.,   # in degrees
        ## plot
        plot = False,
        ## test
        quick_test = False,
        quick_test_file_duration = 5, # in seconds
        ## overwrite files
        overwrite = False,
    )
    params['window_overlap'] = params['window_size'] / 2
    params['nfft'] = params['window_size']
    params['median_filter_vicinity_radius'] = params['doa_std_vicinity_radius']

    # ########### User defined parameters ##############
    if preset_string is None or preset_string == '0':
        params['preset_string'] = '0'
        params['num_min_valid_bins'] = 1
        params['plot'] = True
        params['quick_test'] = True

    #####################################
    elif preset_string == 'A':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.75
        params['r'] = 2
        params['num_min_valid_bins'] = 10
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.75
        params['doa_median_average_nan_th'] = 0.9
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 30
        params['num_init_kmeans'] = 4

    elif preset_string == 'A_eval':
        params['mode'] = 'eval'
        params['preset_string'] = 'A'
        params['directivity_th'] = 0.75
        params['r'] = 2
        params['num_min_valid_bins'] = 10
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.75
        params['doa_median_average_nan_th'] = 0.9
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 30
        params['num_init_kmeans'] = 4
        params['mode'] = 'eval'

    #####################################
    elif preset_string == 'B':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [5,10]

    elif preset_string == 'B_eval':
        params['mode'] = 'eval'
        params['preset_string'] = 'B'
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [5,10]

    #####################################
    elif preset_string == 'Q':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 10
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5
        params['energy_density_local_th_size'] = 11

    elif preset_string == 'Q_eval':
        params['mode'] = 'eval'
        params['preset_string'] = 'Q'
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 10
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5
        params['energy_density_local_th_size'] = 11
    elif preset_string == '8_r1':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.75
        params['r'] = 1
        params['num_min_valid_bins'] = 10
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.75
        params['doa_median_average_nan_th'] = 0.9
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 30


    #####################################
    # Some random trials...
    elif preset_string == '8_r4':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.75
        params['r'] = 4
        params['num_min_valid_bins'] = 10
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.75
        params['doa_median_average_nan_th'] = 0.9
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 30

    elif preset_string == '9c':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 4
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4

    elif preset_string == '9c_r2':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 2
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4

    elif preset_string == '9c_r6':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 6
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4

    elif preset_string == '9c_r8':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 8
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4

    elif preset_string == '9c_r10':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        # mv1 = 3, sv1 = 3

    elif preset_string == '9c_r10_mv1_sv1':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = 1
        # params['std_doa_vicinity_radius'] = 1
    elif preset_string == '9c_r10_mv15_sv1':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [1,5]
        # params['std_doa_vicinity_radius'] = 1
    elif preset_string == '9c_r10_mv5_sv5':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = 5
        # params['std_doa_vicinity_radius'] = 3

    elif preset_string == 'B_short':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 4
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [5, 10]

    elif preset_string == 'B_short2':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [10, 10]
        params['std_doa_vicinity_radius'] = 5

    elif preset_string == 'B_short3':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [10, 10]
        params['std_doa_vicinity_radius'] = 5

    elif preset_string == 'B_short4':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5

    elif preset_string == 'B_short4q':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5

    elif preset_string == 'B_short4q_2':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 10
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5


    elif preset_string == 'B_short4q_2zzz':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 10
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5
        params['energy_density_local_th_size'] = 1


    elif preset_string == 'B_short4q_2www':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 10
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5
        params['energy_density_local_th_size'] = 11

    elif preset_string == 'B_short4_low':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5
        params['energy_density_local_th_size'] = 21

    elif preset_string == 'B_short4_high':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 10
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.85
        params['doa_median_average_nan_th'] = 0.5
        params['plot'] = False
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 8000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['min_num_frames_per_event'] = 8
        params['overwrite'] = True
        params['num_init_kmeans'] = 4
        params['median_filter_vicinity_radius'] = [20, 20]
        params['std_doa_vicinity_radius'] = 5
        params['energy_density_local_th_size'] = 91

    elif preset_string == 'C':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.5
        params['r'] = 4
        params['num_min_valid_bins'] = 1
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.5
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 10  # in seconds'
        params['min_std_overlapping'] = 20
        params['overwrite'] = False
        params['min_num_frames_per_event'] = 4

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    print('-------------- GET PARAMETERS --------------')
    print('Preset: '+preset_string)

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
