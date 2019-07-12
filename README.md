## A HYBRID PARAMETRIC-DEEP LEARNING APPROACH FOR SOUND EVENT LOCALIZATION AND DETECTION

This repository contains the code corresponding to the following paper. If you use this code or part of it, please cite:

> Andres Perez-Lopez, Eduardo Fonseca, Xavier Serra, "A Hybrid Parametric-Deep Learning Approach for Sound Event Localization and Detection", Submitted to DCASE2019 Challenge.


Introduction
-----------------
The method implemented represents a novel approach for the Sound Event Localization and Detection (SELD) task, which is Task 3 of DCASE2019 Challenge.
We use the TAU Spatial Sound Events 2019 - Ambisonic dataset, which provides First-Order Ambisonic (FOA) recordings.
For more details about the task setup and dataset, please check the corresponding [DCASE website](dcase.community/challenge2019/task-sound-event-localization-and-detection).

The method implemented is based on four systems: DOA estimation, association, beamforming and classification, as shown in the following Figure. In turn, the three former systems are conceptually grouped into the so called  _frontend_, while the classification conforms the _backend_. 

<p align="center">
<img src="/figs/system_arch.png" alt="system architecture" width="500"/>
</p>

 - **parametric frontend**:
   1. DOA estimation: The input data is preprocessed by a parametric spatial audio analysis, which yields time-frequency DOA estimations. An schematic representation of the method is shown in the following Figure.
	<p align="center">
	<img src="/figs/doa.png" alt="DOA system" width="700"/>
	</p>
   2. Association: The spatial-temporal information is grouped into _events_, each of them having specific onset/offset times. The next Figure depicts the algorithms.
	<p align="center">
	<br>
	<img src="/figs/association2.png" alt="association system" width="700"/>
	</p>
   3. Beamforming: Given the angular position and the temporal context, each event is mono-segmentated by beamforming in the input ambisonic scene.

 - **classification backend**: The estimated event signals are finally labelled by a multi-class CRNN, illustrated in the next Figure.

<p align="center">
<img src="/figs/DCASE19Task3_backend_archi_v3.png" alt="classification backend" width="500"/>
</p>

Please, refer to the publication for a more detailed explanation of the method, including evaluation metrics and discussion.


Implementation and Usage
-----------------
The SELD task is internally implemented in two different stages.
First, `compute_doa.py` estimates DOAs and audio segments from the dataset, and writes the results as csv files.
The same code is used for both development and evaluation sets.

The signals estimated by the frontend are organized as an intermediate dataset of monophonic sound events (although some leakage is expected in some of them).
Two versions of this intermediate dataset must be computed:

 - using an **ideal frontend**. Audio clips are obtained using the ground truth DOAs and onset/offset times as inputs to the
beamformer; hence beamforming is the only processing carried out. Use `preprocess_metadata_files.py`for this. Files are stored in `data/mono_data/wav/dev/`

 - using the **proposed complete frontend**. Use `process_doa_results.py` for this. Files are stored in `data/mono_data/wav/dev_param_Q/`

These scripts need that the development set (400 wav files of TAU Spatial Sound Events 2019 - Ambisonic) be placed in `data/foa_dev/`, and the accompanying metadata be placed in `/data/metadata_dev`.
The outcome of scripts `preprocess_metadata_files.py` and `process_doa_results.py` is:
 - the deterministic list of audio clips stored, and
 - the corresponding ground truth csv files to access the clips subsequently (namely `gt_dev.csv` and `gt_dev_parametric_Q.csv`, which must be placed in `data/mono_data` for further processing.

During _development_, the classification backend is trained using the provided four fold cross-validation setup in the **development set**.
The CRNN is trained always on the outcome of the ideal frontend.
The CRNN is tested on both the outcome of the ideal frontend and proposed complete frontend.
See `classif/classify.py` for more details. You can run it with

`CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=tensorflow python classify.py -p params.yaml &> output_logs.out`

During _evaluation mode_ (ie, challenge submission), the CRNN is trained on the entire development set processed by the ideal frontend.
Then, we predict on the evaluation set previously processed by the proposed complete frontend.


Results are written in two different csv conventions: _metadata_ and _output_.
_Metadata_ files follow the grountruth convention: each row corresponds to a sound event, 
and information about class, onset/offset time (in seconds), and angular position is provided.
Conversely, _output_ files are frame-based (with a row for each frame with activity),
and thus the onset/offset information is not explicitly stated.

In evaluation mode, DOA estimations are written in the corresponding results folder (`results_metadata` or `results_output`),
within the `/doa` folder. Those first annotations do not contain the classification estimation.
Then, after running the deep learning classification backend, the final result files are written into
the `/classif` folder. 
Therefore, the result files following the required challenge format are stored in
`results_ouput/[DATASET_MODE_PRESET]/classif`.

Results reported in the paper correspond to the configuration "Q".
Therefore, the result files can be found in `results_ouput/foa_[MODE]_Q]/classif`).


Project Structure
-----------------

#### /root
Code for DOA and utils.

`compute_doa.py`
Main DOA estimation loop. Based on the provided parameter configuration, iterates over the files
computing localization and segmentation. As the process output, result files (in both output and metadata formats)
are writen in the corresponding folders. 

`compute_doa_metrics.py`
Script to compute DOA metrics on the analysis results.
Adapted from `seld_dcase2019_master/calculate_SELD_metrics.py`.

`doa_methods.py`
This file contains the source localization and segmentation functions.

`file_utils.py`
Some handy functions to read/write files in the required formats.

`parameters.py`
The different run configurations are described here as a dictionary.
There is only one method described, `get_params(preset_string=None)`, which
defines all possible values that  parameters can take when running the 
algorithm.
Different configuration presets can be identified by the `preset_string`.

`preprocess_metadata_files.py`
This file contains the code to extract monophonic estimates of the sources
given the development dataset and the groundtruth. Beamforming is the only processing carried out (ideal frontend).
Files are stored in `data/mono_data/wav/dev/`

`process_doa_results.py`
This file contains the code to extract monophonic estimates of the sources
given the development dataset WITHOUT any groundtruth. The complete proposed frontend is applied.
Files are stored in `data/mono_data/wav/dev_param_Q/`

`README.md`
This file.

`utils.py`
Some convenience mathematical functions and classes.

`visualize_output.py`
Script to visualize DOA metrics on the analysis results.
Adapted from `seld_dcase2019_master/misc_files/visualize_SELD_output.py`.

#### /data
Folder where the dataset should be located. 
Due to the size, the contents of this folder are not included in git.

#### /results_output
The algorithm output is stored here, in form of .csv files, as in the required challenge output format.
Each folder corresponds to the output with a different dataset type, mode and
configuration preset. Within each folder, folder `doa` contains localization and timing estimations,
and `classif` adds the source classification.

#### /results_metadata
The doa-estimation output in human-readable style, formatted for the source classification step.
Each folder corresponds to the output with a different dataset type, mode and
configuration preset. Within each folder, folder `doa` contains localization and timing estimations,
and `classif` adds the source classification.

#### /seld_dcase2019_master
This is just a fork of the baseline method, as of 28th March 2019 
(commit c11188984875600a607d85f98ca05958ad9287ab).
Some of the methods (evaluation, plot) are taken from there. 
It should be also possible to train and run it...


#### /classif
Deep learning classification backend code.

`classify.py` is the main script. After training and testing the CRNN using multi-class classification accuracy, SELD metrics are computed using first the proposed complete fronten, and also using the ideal forntend  
`data.py` contains the data generators and mixup code  
`feat_extract.py` contains feature extraction code  
`architectures.py` contains the CRNN architecture  
`utils_classif.py` some basic utilities  
`eval.py` evaluation code (only for computing accuracy in the multi-class classsification problem during training; nothing to do with SELD metrics)

Dependencies
-----------------
 - [Parametric spatial audio processing](https://github.com/andresperezlopez/parametric_spatial_audio_processing/tree/master)
