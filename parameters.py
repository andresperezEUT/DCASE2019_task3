## Obviously inspired by seld_dcase2019_master.parameter.py

def get_params(preset_string=None):

    # ########### default parameters ##############
    params = dict(

        preset_string = '0',

        # Base folder containing the foa/mic and metadata folders
        dataset_dir = './data',

        # Folder for saving final results
        output_result_folder_path = './results_output',
        output_result_file_extension = '.csv',

        # Folder for saving intermediate metadata results
        metadata_result_folder_path='./results_metadata',
        metadata_result_file_extension='.csv',

        # Name for after classification folders
        before_classification_folder_name='doa',
        after_classification_folder_name='classif',

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',      # 'foa' - ambisonic or 'mic' - microphone signals

        # MODEL PARAMETERS
        doa_method ='doa_method_mean',
        default_class_id = -1,
        file_duration = 60, # in seconds
        window_size = 256,
        fmin = 125,
        fmax = 8000,
        r = 4,
        directivity_th = 0.75,
        doa_std_vicinity_radius = 3,
        doa_std_th = 0.75,
        num_min_valid_bins = 10,
        required_window_hop = 0.02,
        kmeans_rate_th = 20,


        # PLOT
        plot = False,

        # TEST
        quick_test = False,
        quick_test_file_duration = 0.75, # in seconds
    )
    params['window_overlap'] = params['window_size'] / 2
    params['nfft'] = params['window_size']

    # ########### User defined parameters ##############
    if preset_string is None or preset_string == '0':
        params['preset_string'] = '0'
        params['num_min_valid_bins'] = 1
        params['plot'] = True
        params['quick_test'] = True

    elif preset_string == '1':
        params['preset_string'] = preset_string
        params['directivity_th'] = 0.9
        params['num_min_valid_bins'] = 1
        params['plot'] = True
        params['quick_test'] = True

    elif preset_string == '2':
        params['preset_string'] = preset_string
        params['doa_method'] = 'doa_method_median'

    elif preset_string == '3':
        params['preset_string'] = preset_string
        params['doa_method'] = 'doa_method_median'
        params['directivity_th'] = 0.9

    elif preset_string == '4':
        params['preset_string'] = preset_string
        params['doa_method'] = 'doa_method_variance'
        params['directivity_th'] = 0.75
        params['num_min_valid_bins'] = 5
        params['doa_std_vicinity_radius'] = [5,2]
        params['doa_std_th'] = 0.75
        params['plot'] = False
        params['quick_test'] = False

    elif preset_string == '5':
        params['preset_string'] = preset_string
        params['doa_method'] = 'doa_method_variance'
        params['directivity_th'] = 0.75
        params['num_min_valid_bins'] = 5
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.75
        params['plot'] = True
        params['quick_test'] = True

    elif preset_string == '6':
        params['preset_string'] = preset_string
        params['doa_method'] = 'doa_method_variance'
        params['directivity_th'] = 0.95
        params['r'] = 0
        params['num_min_valid_bins'] = 10
        params['doa_std_vicinity_radius'] = 10
        params['doa_std_th'] = 0.5
        params['kmeans_rate_th'] = 1
        params['doa_median_average_nan_th'] = 0.75
        params['plot'] = True
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 5 # in seconds'

    elif preset_string == '7':
        params['preset_string'] = preset_string
        params['doa_method'] = 'doa_method_variance'
        params['directivity_th'] = 0.95
        params['r'] = 2
        params['num_min_valid_bins'] = 10
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.95
        params['kmeans_rate_th'] = 1
        params['doa_median_average_nan_th'] = 0.9
        params['plot'] = True
        params['quick_test'] = True
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 5 # in seconds'
        params['min_std_overlapping'] = 30

    elif preset_string == '8':
        params['preset_string'] = preset_string
        params['doa_method'] = 'doa_method_variance'
        params['directivity_th'] = 0.75
        params['r'] = 2
        params['num_min_valid_bins'] = 10
        params['doa_std_vicinity_radius'] = 2
        params['doa_std_th'] = 0.75
        params['kmeans_rate_th'] = 1
        params['doa_median_average_nan_th'] = 0.9
        params['plot'] = False
        params['quick_test'] = False
        params['fmin'] = 0
        params['fmax'] = 15000
        params['quick_test_file_duration'] = 5  # in seconds'
        params['min_std_overlapping'] = 30


    # elif argv == '3':
    #     params['mode'] = 'eval'
    #     params['dataset'] = 'mic'
    #
    # elif argv == '4':
    #     params['mode'] = 'dev'
    #     params['dataset'] = 'foa'
    #
    # elif argv == '5':
    #     params['mode'] = 'eval'
    #     params['dataset'] = 'foa'
    #
    # # Quick test
    # elif argv == '999':
    #     print("QUICK TEST MODE\n")
    #     params['quick_test'] = True
    #     params['epochs_per_fit'] = 1
    #
    # else:
    #     print('ERROR: unknown argument {}'.format(argv))
    #     exit()

    print('-------------- GET PARAMETERS --------------')
    print('Preset: '+preset_string)

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
