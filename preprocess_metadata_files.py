from parameters import get_params
from utils import beamforming

import os.path
import soundfile as sf
import csv
import numpy as np
import matplotlib.pyplot as plt

# new imports
import pandas as pd

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

preset = '0' # Default preset
params = get_params(preset)

# Get path to data to be processed
dataset_dir = params['dataset_dir']
dataset_type_folder = params['dataset'] + "_" + params['mode']
data_folder_path = os.path.join(dataset_dir, dataset_type_folder)

# Get path to groundtruth metadata
gt_folder = 'metadata' + "_" + params['mode']
gt_folder_path = os.path.join(dataset_dir, gt_folder)

# store ground truth info for classification
files =[]
labels = []
splits = []
irs = []
parents = []

# Iterate over all audio files
audio_files = [f for f in os.listdir(data_folder_path) if not f.startswith('.')]
for audio_file_name in audio_files:

    # Open audio file
    b_format, sr = sf.read(os.path.join(data_folder_path, audio_file_name))

    # Get associated metadata file
    metadata_file_name = os.path.splitext(audio_file_name)[0] + params['metadata_result_file_extension']

    # Iterate over the associated metadata file
    with open(os.path.join(gt_folder_path,metadata_file_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Discard the first line (just the column titles)
            if i > 0 :

                # Get values for this sound event
                sound_class_string = row[0]
                start_time_seconds = float(row[1])
                end_time_seconds = float(row[2])
                elevation = float(row[3])
                azimuth = float(row[4])
                distance = float(row[5])

                # Slice the b_format audio to the corresponding event length
                start_frame = int(np.floor(start_time_seconds * sr ))
                end_frame = int(np.ceil(end_time_seconds * sr ))

                # Steer a beam and estimate the source
                beamforming_method = 'basic'
                # You can try also with 'inphase', I would say there is not much difference in the non-overlapping case...
                sound_event_mono = beamforming(b_format[start_frame:end_frame], azimuth, elevation, beamforming_method)

                # Write audio to file
                filename = sound_class_string + '_' + str(start_frame) + '_' + metadata_file_name.split('.')[0] + '.wav'
                path_to_write = os.path.join('data/mono_data/wav/dev')
                if not os.path.exists(path_to_write):
                    os.mkdir(path_to_write)
                sf.write(os.path.join(path_to_write, filename), sound_event_mono, sr)

                # create csv with split info for development
                files.append(filename)
                labels.append(sound_class_string)
                splits.append(int(metadata_file_name.split('_')[0][-1]))
                irs.append(int(metadata_file_name.split('_')[1][-1]))
                parents.append(metadata_file_name)


gt_classif = pd.DataFrame(files, columns=['fname'])
gt_classif['label'] = labels
gt_classif['split'] = splits
gt_classif['ir'] = irs
gt_classif['parent'] = parents
gt_classif.to_csv('gt_dev.csv', index=False)
print('EOF')
# plt.show()



