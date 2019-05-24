import os.path
import soundfile as sf

from parameters import get_params
import doa_methods
from compute_doa_metrics import compute_DOA_metrics
from visualize_output import visualize_output
from file_utils import write_output_result_file, write_metadata_result_file

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAIN

# SELECT THE PARAMETER PRESET HERE!
preset = '6'
params = get_params(preset)

# Get path to data to be processed
dataset_dir = params['dataset_dir']
dataset_type_folder = params['dataset'] + "_" + params['mode']
data_folder_path = os.path.join(dataset_dir, dataset_type_folder)

# Check if result folders exist, and create them in turn
output_result_folder_path = os.path.join(params['output_result_folder_path'], dataset_type_folder + "_" + preset)
if not os.path.exists(output_result_folder_path):
    os.mkdir(output_result_folder_path)
metadata_result_folder_path = os.path.join(params['metadata_result_folder_path'], dataset_type_folder + "_" + preset)
if not os.path.exists(metadata_result_folder_path):
    os.mkdir(metadata_result_folder_path)

print('                                              ')
print('-------------- PROCESSING FILES --------------')
print('                                              ')
print('Folder path: ' + data_folder_path              )


# Iterate over all audio files
for audio_file in os.listdir(data_folder_path):
    if audio_file != '.DS_Store': # Fucking OSX

        # Open audio file
        b_format, sr = sf.read(os.path.join(data_folder_path,audio_file))

        # -------------- DOA ESTIMATION --------------
        # Perform DOA estimation
        doa_method_instance = getattr(doa_methods, params['doa_method'])
        result, result_quantized = doa_method_instance(b_format, sr, params)

        # Postprocess the localization results
        # TODO!
        # metadata_result_array = doa_methods.group_sources(result_averaged_dict)
        # metadata_result_array, result_averaged_dict = doa_methods.group_sources_q(result_quantized, params)
        metadata_result_array, result_averaged_dict = doa_methods.group_sources_q_overlap(result_quantized, params)

        # Write the localization results in the metadata format
        # TODO!
        # metadata_result_file_name = os.path.splitext(audio_file)[0] + params['metadata_result_file_extension']
        # metadata_result_file_path = os.path.join(metadata_result_folder_path, metadata_result_file_name)
        # write_metadata_result_file(metadata_result_array, metadata_result_file_path)

        # -------------- SOURCE CLASSIFICATION --------------
        # Extract mono sources from metadata result file and b-format audio
        # beamforming(b_format,)


        # Write DOA output results to file in the proper format

        # TODO: PUT INSIDE PARAMETERS
        # ## WRITE FROM AVERAGED DICT
        # output_result_file_name = os.path.splitext(audio_file)[0] + params['output_result_file_extension']
        # output_result_file_path = os.path.join(output_result_folder_path, output_result_file_name)
        # write_output_result_file(result_averaged_dict, output_result_file_path)

        output_result_file_name = os.path.splitext(audio_file)[0] + params['output_result_file_extension']
        output_result_file_path = os.path.join(output_result_folder_path, output_result_file_name)
        write_output_result_file(result_averaged_dict, output_result_file_path)

print('-------------- PROCESSING FINISHED --------------')
print('                                                 ')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('-------------- COMPUTE DOA METRICS --------------')

gt_folder = os.path.join(dataset_dir, 'metadata_'+params['mode'])
compute_DOA_metrics(gt_folder, output_result_folder_path)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('-------------- VISUALIZE OUTPUT --------------')

visualize_output(output_result_folder_path, gt_folder, data_folder_path, params)


