DCASE2019 task 3 
======

Introduction
-----------------


Project Structure
-----------------

#### /root

##### compute_doa_metrics.py
TODO

##### doa_methods.py
This file contains the source localization functions.

##### file_utils.py
Some handy functions to read/write files in the required format.

##### parameters.py
The different run configurations are described here as a dictionary.
There is only one method described, `get_params(preset_string=None)`, which
defines all possible values that  parameters can take when running the 
algorithm.
Different configuration presets can be identified by the `preset_string`,
incuding a default configuration.

##### Preprocess_metadata_files.py
This file contains the code to extract monophonic estimates of the sources
given the groundtruth.

##### process.py
Main process.
TODO

##### README.md
This file.

##### requirements.txt
TODO

##### utils.txt
Some convenience functions and classes.

##### visualize_output.py
TODO

#### /data
Folder where the development data is located. 
Due to the size, the contents of this folder are not included in git.

#### /models
We can save the models here

#### /notebooks
If we want to produce some fancy notebooks for the conference, etc.

#### /results_output
The algorithm output is stored here, in form of .csv files.
Each folder corresponds to the output with a different dataset type, mode and
configuration preset.

#### /results_metadata
The doa-estimation output in human-readable format.
Also prepared for the source classification step.
Each folder corresponds to the output with a different dataset type, mode and
configuration preset.

#### /seld_dcase2019_master
This is just a fork of the baseline method, as of 28th March 2019 
(commit c11188984875600a607d85f98ca05958ad9287ab).
Some of the methods (evaluation, plot) are taken from there. 
It should be also possible to train and run it...