## A HYBRID PARAMETRIC-DEEP LEARNING APPROACH FOR SOUND EVENT LOCALIZATION AND DETECTION
======

This repository contains the code corresponding to the following paper. If you use this code or part of it, please cite:

> Andres Perez-Lopez, Eduardo Fonseca, Xavier Serra, "A HYBRID PARAMETRIC-DEEP LEARNING APPROACH FOR SOUND EVENT LOCALIZATION AND DETECTION",
Submitted to DCASE2019 Challenge.

We are currently working on cleaning the code, and making it usable by third parties,
so that the results can be reproduced. Thanks for being patient!


Introduction
-----------------
The method implemented here represents a novel approach for the SELD task based on three different building blocks.
First, the input data is preprocessed by a parametric spatial audio analysis, which yields time-frequency DOA estimations.
Then, the spatial-temporal information is grouped into _events_, each of them having a specific onset/offset time.
Given the angular position and the temporal context, each event is mono-segmentated by beamforming in the input ambisonic scene.
The estimated signals are finally labelled by a multi-class CRNN. (TODO 4EDU revisar).
Figure 1 shows a general overview of the method.


![alt text](https://github.com/andresperezlopez/DCASE2019_task3/tree/master/figs/DCASE19Task3_backend_archi_v3.png "Logo Title Text 1")



Please, refer to the publication for a more detailed explanation of the method,
including evaluation metrics and discussion.


Implementation and Usage
-----------------
The SELD task is internally implemented in two different stages.
First, `compute_doa.py` estimates DOAs and audio segments from the dataset, and writes the results as csv files.
The same code is used for both development and evaluation sets. 
Then, (TODO 4EDU I don't know the details. Write in general terms about the classes/methods used for devel/eval).

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


##### compute_doa.py
Main DOA estimation loop. Based on the provided parameter configuration, iterates over the files
computing localization and segmentation. As the process output, result files (in both output and metadata formats)
are writen in the corresponding folders. 

##### compute_doa_metrics.py
Script to compute DOA metrics on the analysis results.
Adapted from `seld_dcase2019_master/calculate_SELD_metrics.py`.

##### doa_methods.py
This file contains the source localization and segmentation functions.

##### file_utils.py
Some handy functions to read/write files in the required formats.

##### parameters.py
The different run configurations are described here as a dictionary.
There is only one method described, `get_params(preset_string=None)`, which
defines all possible values that  parameters can take when running the 
algorithm.
Different configuration presets can be identified by the `preset_string`.

##### Preprocess_metadata_files.py
This file contains the code to extract monophonic estimates of the sources
given the development dataset and the groundtruth.

##### process_doa_results.py
TODO 4EDU, do you use that?

##### README.md
This file.

##### recognition.py
TODO 4EDU, do you use that? probably not

##### utils.txt
Some convenience mathematical functions and classes.

##### visualize_output.py
Script to visualize DOA metrics on the analysis results.
Adapted from `seld_dcase2019_master/misc_files/visualize_SELD_output.py`.

#### /data
Folder where the dataset should be located. 
Due to the size, the contents of this folder are not included in git.

#### /models
We can save the models here

#### /notebooks
If we want to produce some fancy notebooks for the conference, etc.

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
TODO 4EDU


Dependencies
-----------------
 - [Parametric spatial audio processing](https://github.com/andresperezlopez/parametric_spatial_audio_processing/tree/master)