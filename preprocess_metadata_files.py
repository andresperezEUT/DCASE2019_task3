from parameters import get_params
from utils import beamforming

import os.path
import soundfile as sf
import csv
import numpy as np
import matplotlib.pyplot as plt

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

# Iterate over all audio files
for audio_file_name in os.listdir(data_folder_path):
    if audio_file_name != '.DS_Store': # Fucking OSX

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

                    # Here you go!
                    plt.figure()
                    plt.suptitle(sound_class_string)
                    plt.plot(sound_event_mono)
                    plt.grid()

                    sf.write('/Users/andres.perez/Desktop/sources/'+sound_class_string+str(start_frame)+'.wav',sound_event_mono,sr)




plt.show()



