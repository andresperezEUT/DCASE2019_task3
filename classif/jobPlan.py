import yaml
import numpy as np
import time
import subprocess
import sys

"""
script to schedule a plan of jobs for a specific model, including:
-N trials of the experiment (to average scores and report more consistently)
-trying with some different parameters for analysis

from this script, we edit a config/params_file.yaml, and call a main script
Then, the main script loads all the params from yaml and runs the experiment
"""

# ---------
start = time.time()

# nb of trials in the experiment
N = 3

# params to try************************************
lrs = [0.001, 0.0001]
# patch_lens = [25, 50, 75, 100]
# patch_lens = [25, 50]
# output_file = 'js_tidy_patch_len2550'
# models = ['js_tidy']

# output_file = 'vgg_md_patch_len2550'
# models = ['vgg_md']

# output_file = 'crnn_patch_len2550'
# models = ['crnn']

patch_lens = [35, 50]
output_file = 'mobilenet_patch_len3550'
models = ['mobileKERAS']

losses = ['CCE']  # CCE_diy_max, lq_loss, CCE_diy_outlier, CCE, CCE_diy_max_origin, CCE_diy_outlier_origin, lq_loss_origin
# losses = ['lq_loss_origin']  # CCE_diy_max, lq_loss, CCE_diy_outlier, CCE, CCE_diy_max_origin, CCE_diy_outlier_origin, lq_loss_origin
# q_losses = [0.5, 0.6]
# q_losses = [0.4, 0.7]
# q_losses = [0.5, 0.7]
# q_losses = [0.7]

# vip --------------------------------dummy run for debugging
# proto = 'proto5'                    # this is the dataset version, currently prototype 5. this should be fixed.
# output_file = 'out_eval'            # this the name corresponding to the experiment. The log file (and other outcome files have this string for easy identification)
# models = ['debug']                  # for this experiments we choose either 'kong' (VGG style 8 layers) and 'js_tidy' (CNN 3 layers). 'debug' is a dummy model for fast debugging
# losses = ['CCE_diy_max']                    # define the loss function. CCE means categorical cross_entropy,
# # q_losses = [0.99]                 # this is an example of a hyper-parameter to test
# # q_losses = [0.9, 0.8, 0.6, 0.5]     # this is an example of a hyper-parameter to test with several values (typical usage)
# reed_betas = [0.9]     # this is an example of a hyper-parameter to test with several values (typical usage)
# m_losses = [0.8]

# vip define the path of the yaml with (most) of the parameters that define the experiment
yaml_file = 'params_edu_v1.yaml'


def change_yaml(fname, count_trial, output_file, model, loss, patch_len, lr):
    """
    Modifies the yaml fiven by fname according to the input parameters.
    This allows to test several values for hyper-parameter(s) on the same run
    :param fname:
    :param count_trial:
    :param train_data:
    :param output_file:
    :param model:
    :param loss:
    :param q_loss:
    :return:
    """

    stream = open(fname, 'r')
    data = yaml.load(stream)

    data['ctrl']['count_trial'] = count_trial
    data['ctrl']['output_file'] = output_file
    data['learn']['model'] = model
    data['loss']['type'] = loss

    # watch basic
    data['learn']['lr'] = lr
    data['extract']['patch_len'] = patch_len

    # watch LSR
    data['learn']['LSR'] = False
    data['learn']['LSRmode'] = False    # GROUPS2 GROUPS3 WEIGHTS_CC WEIGHTS_CC+
    data['learn']['eps_LSR_noisy'] = False       # False, epsi
    data['learn']['distri_prior_LSR'] = False    # False=uniform, 'noise', 'unigram'
    data['learn']['delta_k_LSR'] = 38            # 2,4,6,8, only for noise distri

    data['learn']['delta_eps_LSR'] = False      # this is concurrent to all distris. put to False if undesired
    data['learn']['LSRmapping'] = False         # LSRmapping False

    # watch 2 stage learning***********************, just define the minimum to ignore
    data['learn']['stages'] = 1                  #usually 1 or 2. but 0 for enabling mixup_warmup

    # watch 1 stage dropout***********************
    data['learn']['dropout'] = False                    # True False
    data['learn']['dropout_prob'] = False       # only used if True

    # watch 1 stage mixup***********************
    data['learn']['mixup'] = False                    # True False
    data['learn']['mixup_mode'] = False
    data['learn']['mixup_alpha'] = False
    data['learn']['mixup_log'] = False
    data['learn']['mixup_clamp'] = False
    data['learn']['mixup_warmup_epochs'] = False

    data['learn']['early_stop'] = 'val_acc'        # True False

    with open(fname, 'w') as yaml_file:
        yaml_file.write(yaml.dump(data, default_flow_style=False))


def main():

    # to append suffix and store some results for every trial of the experiment (usually 6 trials)
    count_trial = 0

    for kk in np.arange(N):
        # consistency
        for model in models:
            for loss in losses:
                # for q_loss in q_losses:
                for lr in lrs:
                    for patch_len in patch_lens:
                        count_trial += 1

                        change_yaml(yaml_file,
                                    count_trial=count_trial,
                                    output_file=output_file,
                                    model=model,
                                    loss=loss,
                                    patch_len=patch_len,
                                    lr=lr
                                    )

                        # call the job
                        str_exec = 'python classify.py -p ' + yaml_file
                        print(str_exec)
                        # CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=tensorflow python jobPlan.py &> logs/pro4_model_set_approach_hyper.out

                        try:
                            retcode = subprocess.call(str_exec, shell=True)
                            if retcode < 0:
                                print("Child was terminated by signal", -retcode, file=sys.stderr)
                            else:
                                print("Child returned", retcode, file=sys.stderr)
                        except OSError as e:
                            print("Execution failed:", e, file=sys.stderr)

    end = time.time()

    str_result_generate = "sed -n -e 's/^.*Mean Accuracy for files evaluated: //p' logs/" + output_file + ".out | tr '\n' ',' > logs/" + output_file + ".csv"

    retcode_res = subprocess.call(str_result_generate, shell=True)
    if retcode_res < 0:
        print("Child was terminated by signal", -retcode_res, file=sys.stderr)
    else:
        print("Child returned", retcode_res, file=sys.stderr)

    #
    print('\n=============================List of jobs finalized=================================================\n')
    print('\nTime elapsed for the LIST of jobs: %7.2f hours' % ((end - start) / 3600.0))
    print('\n====================================================================================================\n')

    return 0


if __name__ == '__main__':
    main()
