import os

from compute_doa_metrics import compute_DOA_metrics
from file_utils import write_metadata_result_file, build_result_dict_from_metadata_array, write_output_result_file
from parameters import get_params
import soundfile as sf
import csv
import numpy as np
from utils import beamforming, dummy_classifier
from visualize_output import visualize_output
import pandas as pd

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Path stuff

# This parameter will define the algorithm type
preset_string = 'Q'

# Default preset: contains path to folders
params = get_params(preset_string)

# Dataset type:
dataset_type_folder = params['dataset'] + "_" + params['mode']
dataset_preset_folder = dataset_type_folder + '_' + params['preset_string']

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
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]

# store ground truth info for classification
files =[]
labels = []
splits = []
irs = []
parents = []

for audio_file_name in audio_files:

    # Open audio file
    b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))

    # Get associated metadata file
    metadata_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']

    # This is our modified metadata result array
    metadata_result_classif_array = []

    # Iterate over the associated doa metadata file
    with open(os.path.join(results_metadata_doa_folder, metadata_file_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        counter = 0
        for i, row in enumerate(reader):
            # Discard the first line (just the column titles)
            if i > 0:
                # Get values for this sound event
                if params['mode'] == 'eval':
                    sound_class_string = 'eval_mode'
                else:
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
                if params['mode'] == 'eval':
                    filename = sound_class_string + '_' + str(start_frame) + '_' + str(end_frame) + '_' + metadata_file_name.split('.')[0] + '_c' + str(counter) + '.wav'
                else:
                    filename = sound_class_string + '_' + str(start_frame) + '_' + str(end_frame) + '_' + metadata_file_name.split('.')[0] + '.wav'

                # TODO 4EDU: aqui he cambiado para que escriba en 'data/mono_data/wav/dev_param'
                # si eso es lo que hay que hacer en eval...
                # path_to_write = os.path.join('data/mono_data/wav', params['mode'] + '_param_2' + preset_string )
                path_to_write = os.path.join('data/mono_data/wav', params['mode'] + '_param_' + preset_string )
                if not os.path.exists(path_to_write):
                    os.mkdir(path_to_write)
                sf.write(os.path.join(path_to_write, filename), sound_event_mono, sr)

                # create csv with split info for development and test based on parametric frontend
                files.append(filename)
                labels.append(sound_class_string)
                splits.append(int(metadata_file_name.split('_')[0][-1]))
                # irs.append(int(metadata_file_name.split('_')[1][-1]))
                parents.append(metadata_file_name)

                # vip Classify: this will need 4 models for 4 test splits in x-val in development mode
                # PLUS one model for evaluation mode
                class_id = dummy_classifier(sound_event_mono)

                # Substitute the None for the current class, and append to the new metadata array
                row[0] = class_id
                metadata_result_classif_array.append(row)

                # increase event counter within the same recording. this ensures different filenames for the outcome of the frontend
                counter += 1

    # Write a new results_metadata_classif file with the modified classes
    metadata_result_classif_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']
    path_to_write = os.path.join(results_metadata_classif_folder, metadata_result_classif_file_name)
    write_metadata_result_file(metadata_result_classif_array, path_to_write)

    # Write a new result_output_classif file with the modified classes
    output_result_classif_dict = build_result_dict_from_metadata_array(metadata_result_classif_array, params['required_window_hop'])
    path_to_write = os.path.join(results_output_classif_folder, metadata_file_name)
    write_output_result_file(output_result_classif_dict, path_to_write)


# save dataset_dev_mono_parametric
gt_classif = pd.DataFrame(files, columns=['fname'])
gt_classif['label'] = labels
gt_classif['split'] = splits
# gt_classif['ir'] = irs
gt_classif['parent'] = parents

# gt_classif.to_csv('gt_dev_parametric_8.csv', index=False)
# TODO 4EDU: cambiar esta linea por algo como:
gt_csv_file_name = 'gt_' + params['mode'] + '_parametric_' + params['preset_string'] + '.csv'
gt_classif.to_csv(gt_csv_file_name, index=False)

print('EOF')


print('-------------- PROCESSING FINISHED --------------')
print('                                                 ')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('-------------- COMPUTE DOA METRICS --------------')

gt_folder = os.path.join(dataset_dir, 'metadata_'+params['mode'])
compute_DOA_metrics(gt_folder, results_output_classif_folder)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('-------------- VISUALIZE OUTPUT --------------')

visualize_output(results_output_classif_folder, gt_folder, data_folder_path, params)