import os

from compute_doa_metrics import compute_DOA_metrics
from file_utils import write_metadata_result_file, build_result_dict_from_metadata_array, write_output_result_file
from parameters import get_params
import soundfile as sf
import csv
import numpy as np
from utils import beamforming, dummy_classifier
from visualize_output import visualize_output


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Path stuff

# This parameter will define the algorithm type
preset_string = '8'

# Default preset: contains path to folders
params = get_params(preset_string)

# Dataset type:
dataset_type_folder = params['dataset'] + "_" + params['mode']
dataset_preset_folder = dataset_type_folder + '_' + preset_string

# Get folder names before and after classification
doa_folder = params['before_classification_folder_name']
classif_folder = params['after_classification_folder_name']

# Path to results_metadata folder _before classification_; it should exist
results_metadata_doa_folder = os.path.join(params['metadata_result_folder_path'],
                                           dataset_preset_folder,
                                           doa_folder)
if not os.path.exists(results_metadata_doa_folder):
    raise ValueError

# Path to results_metadata folder _before classification_; create it if necessary
results_metadata_classif_folder = os.path.join(params['metadata_result_folder_path'],
                                               dataset_preset_folder,
                                               classif_folder)
if not os.path.exists(results_metadata_classif_folder):
    os.mkdir(results_metadata_classif_folder)

# Path to results_output folder _before classification_; it should exist
results_output_doa_folder = os.path.join(params['output_result_folder_path'],
                                           dataset_preset_folder,
                                           doa_folder)
if not os.path.exists(results_output_doa_folder):
    raise ValueError

# Path to results_output folder _before classification_; create it if necessary
results_output_classif_folder = os.path.join(params['output_result_folder_path'],
                                               dataset_preset_folder,
                                               classif_folder)
if not os.path.exists(results_output_classif_folder):
    os.mkdir(results_output_classif_folder)

# Path to audio folder
dataset_dir = params['dataset_dir']
data_folder_path = os.path.join(dataset_dir, dataset_type_folder)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('                                              ')
print('-------------- PROCESSING FILES --------------')
print('                                              ')
print('Folder path: ' + data_folder_path              )

# Iterate over all audio files
for audio_file_name in os.listdir(data_folder_path):
    if audio_file_name != '.DS_Store': # Fucking OSX

        # Open audio file
        b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))

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
                    elevation = float(row[3])
                    azimuth = float(row[4])
                    distance = row[5]

                    # Slice the b_format audio to the corresponding event length
                    start_frame = int(np.floor(start_time_seconds * sr))
                    end_frame = int(np.ceil(end_time_seconds * sr))

                    # Steer a beam and estimate the source
                    beamforming_method = 'basic'
                    # You can try also with 'inphase', I would say there is not much difference in the non-overlapping case...
                    sound_event_mono = beamforming(b_format[start_frame:end_frame], azimuth, elevation,
                                                   beamforming_method)

                    # Classify
                    class_id = dummy_classifier(sound_event_mono)

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


print('-------------- PROCESSING FINISHED --------------')
print('                                                 ')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('-------------- COMPUTE DOA METRICS --------------')

gt_folder = os.path.join(dataset_dir, 'metadata_'+params['mode'])
compute_DOA_metrics(gt_folder, results_output_classif_folder)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('-------------- VISUALIZE OUTPUT --------------')

visualize_output(results_output_classif_folder, gt_folder, data_folder_path, params)