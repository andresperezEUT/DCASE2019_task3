import os.path
import soundfile as sf

from parameters import get_params
import doa_methods
from compute_doa_metrics import compute_DOA_metrics
from visualize_output import visualize_output
from file_utils import write_output_result_file, write_metadata_result_file, \
    assign_metadata_result_classes_from_groundtruth
import time, datetime

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAIN

start_time = time.time()

# SELECT THE PARAMETER PRESET HERE!
preset = 'B_eval'
params = get_params(preset)

# Get path to data to be processed
dataset_dir = params['dataset_dir']
dataset_type_folder = params['dataset'] + "_" + params['mode']
data_folder_path = os.path.join(dataset_dir, dataset_type_folder)

# Check if output result folders exist, and create them in turn
output_result_folder_path = os.path.join(params['output_result_folder_path'],
                                         dataset_type_folder + "_" +
                                         params['preset_string'])
if not os.path.exists(output_result_folder_path):
    os.mkdir(output_result_folder_path)

output_result_doa_folder_path = os.path.join(output_result_folder_path,
                                            params['before_classification_folder_name'])
if not os.path.exists(output_result_doa_folder_path):
    os.mkdir(output_result_doa_folder_path)

# Check if metadata result folders exist, and create them in turn
metadata_result_folder_path = os.path.join(params['metadata_result_folder_path'],
                                           dataset_type_folder + "_" +
                                           params['preset_string'])
if not os.path.exists(metadata_result_folder_path):
    os.mkdir(metadata_result_folder_path)

metadata_result_doa_folder_path = os.path.join(metadata_result_folder_path,
                                              params['before_classification_folder_name'])
if not os.path.exists(metadata_result_doa_folder_path):
    os.mkdir(metadata_result_doa_folder_path)

# Groundtruth folder
gt_folder = os.path.join(dataset_dir, 'metadata_'+params['mode'])



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


print('                                              ')
print('-------------- PROCESSING FILES --------------')
print('                                              ')
print('Folder path: ' + data_folder_path              )

# Iterate over all audio files
audio_file_idx = 0
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]
for audio_file in audio_files:

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print("{}: {}, {}".format(audio_file_idx, st, audio_file))

    # Only perform analysis if not already existing
    output_result_file_name = os.path.splitext(audio_file)[0] + params['output_result_file_extension']
    output_result_file_path = os.path.join(output_result_doa_folder_path, output_result_file_name)
    if os.path.exists(output_result_file_path) and params['overwrite'] is False:
        continue

    # Open audio file
    b_format, sr = sf.read(os.path.join(data_folder_path,audio_file))

    # Perform DOA estimation
    result_quantized = doa_methods.estimate_doa(b_format, sr, params)

    # Postprocess the localization results
    metadata_result_array, output_result_dict = doa_methods.group_events(result_quantized, params)

    # Write the localization results in the metadata format
    metadata_result_file_name = os.path.splitext(audio_file)[0] + params['metadata_result_file_extension']
    metadata_result_file_path = os.path.join(metadata_result_doa_folder_path, metadata_result_file_name)
    write_metadata_result_file(metadata_result_array, metadata_result_file_path)

    # If development dataset, assign event classes from groundtruth
    if params['mode'] == 'dev':
        time_th = params['gt_time_th']
        dist_th = params['gt_dist_th']
        assign_metadata_result_classes_from_groundtruth(metadata_result_file_name,
                                                        metadata_result_doa_folder_path,
                                                        gt_folder, time_th, dist_th)

    # Write the localization results in the output format
    output_result_file_name = os.path.splitext(audio_file)[0] + params['output_result_file_extension']
    output_result_file_path = os.path.join(output_result_doa_folder_path, output_result_file_name)
    write_output_result_file(output_result_dict, output_result_file_path)

    audio_file_idx += 1

print('-------------- PROCESSING FINISHED --------------')
print('                                                 ')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if params['mode'] == 'dev':
    print('-------------- COMPUTE DOA METRICS --------------')
    compute_DOA_metrics(gt_folder, output_result_doa_folder_path)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# if params['mode'] == 'dev':
#     print('-------------- VISUALIZE OUTPUT --------------')
#     visualize_output(output_result_doa_folder_path, gt_folder, data_folder_path, params)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end_time = time.time()

print('                                               ')
print('-------------- PROCESS COMPLETED --------------')
print('                                               ')
print('Elapsed time: ' + str(end_time-start_time)      )
