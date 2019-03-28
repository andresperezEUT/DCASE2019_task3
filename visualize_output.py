
import os
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
import librosa
import numpy as np

import seld_dcase2019_master.metrics.evaluation_metrics as evaluation_metrics
import seld_dcase2019_master.cls_feature_class as cls_feature_class
from seld_dcase2019_master.misc_files.visualize_SELD_output import collect_classwise_data, plot_func


# %% --------------------------------- PARAMETERS -----------------------------------------

# output format file to visualize
# pred = '/home/adavanne/taitoWorkDir/SELD_DCASE2019/results/999_foa_dev/split0_ir0_ov1_1.csv'
pred_path = '/Users/andres.perez/source/DCASE2019/results/subset/'

# path of reference audio directory for visualizing the spectrogram and description directory for
# visualizing the reference
# Note: The code finds out the audio filename from the predicted filename automatically
ref_dir = '/Volumes/Dinge/DCASE2019_subset/metadata_dev/'
aud_dir = '/Volumes/Dinge/DCASE2019_subset/foa_dev/'




# %% --------------------------------- MAIN SCRIPT STARTS HERE -----------------------------------------

# fixed hoplength of 0.02 seconds for evaluation
hop_s = 0.02

for pred_file in os.listdir(pred_path):
    if pred_file != '.DS_Store': # Fucking OSX

        print(pred_file)

        # check that gt file exists

        # load the predicted output format
        pred = os.path.join(pred_path, pred_file)
        pred_dict = evaluation_metrics.load_output_format_file(pred)

        # load the reference output format
        feat_cls = cls_feature_class.FeatureClass()
        ref_filename = os.path.basename(pred)
        ref_path = os.path.join(ref_dir, ref_filename)

        if not os.path.exists(ref_path):
            print('Metadata file not found: ' + ref_path)
        else:
            ref_desc_dict = feat_cls.read_desc_file(ref_path, in_sec=True)
            ref_dict = evaluation_metrics.description_file_to_output_format(ref_desc_dict, feat_cls.get_classes(), hop_s)

            pred_data = collect_classwise_data(pred_dict)
            ref_data = collect_classwise_data(ref_dict)

            nb_classes = len(feat_cls.get_classes())

            # load the audio and extract spectrogram
            ref_filename = os.path.basename(pred).replace('.csv', '.wav')
            audio, fs = feat_cls._load_audio(os.path.join(aud_dir, ref_filename))
            stft = np.abs(np.squeeze(feat_cls._spectrogram(audio[:, :1])))
            stft = librosa.amplitude_to_db(stft, ref=np.max)

            plot.figure(figsize=plot.figaspect(1 / 2.))
            plot.suptitle(ref_filename)
            gs = gridspec.GridSpec(3, 2)
            ax0 = plot.subplot(gs[0, :]), librosa.display.specshow(stft.T, sr=fs, x_axis='time', y_axis='linear')
            # ax1 = plot.subplot(gs[1, :2]), plot_func(ref_data, hop_s, ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED reference')
            # ax2 = plot.subplot(gs[1, 2:]), plot_func(pred_data, hop_s, ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED predicted')
            ax3 = plot.subplot(gs[1, 0]), plot_func(ref_data, hop_s, ind=2), plot.ylim([-190, 190]), plot.title('Azimuth DOA reference')
            ax4 = plot.subplot(gs[1, 1]), plot_func(pred_data, hop_s, ind=2), plot.ylim([-190, 190]), plot.title('Azimuth DOA predicted')
            ax5 = plot.subplot(gs[2, 0]), plot_func(ref_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA reference')
            ax6 = plot.subplot(gs[2, 1]), plot_func(pred_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA predicted')
            # ax_lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
            ax_lst = [ax0, ax3, ax4, ax5, ax6]
            plot.show()

