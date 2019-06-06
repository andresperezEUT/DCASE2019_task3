## FILE UTILS
import numpy as np
import os

from seld_dcase2019_master.metrics.evaluation_metrics import distance_between_spherical_coordinates_rad
from utils import circmedian, deg2rad
import csv
from copy import copy


def write_output_result_file(output_result_dict, path_to_write):
    """
    (Adapted from metrics.evaluation_metrics.write_output_format_file)
    :param output_result_dict:
    :param path_to_write:
    :return:
    """

    # Ensure ascending order
    frames = output_result_dict.keys()
    try:
        frames.sort()
    except:
        # python3
        sorted(frames)

    _fid = open(path_to_write, 'w')
    for _frame_ind in frames:
        _value = output_result_dict[_frame_ind]
        if len(_value) == 3:
            # this is only one point
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind),  # Frame
                                              int(_value[0]),   # Class idx
                                              int(np.round(float(_value[1]))),  # Azimuth
                                              int(np.round(float(_value[2]))))) # Elevation
        elif len(_value) == 2:
            # two sources
            for _v in _value:
                _fid.write('{},{},{},{}\n'.format(int(_frame_ind),  # Frame
                                                  int(_v[0]),       # Class idx
                                                  int(np.round(float(_v[1]))),  # Azimuth
                                                  int(np.round(float(_v[2]))))) # Elevation
    _fid.close()




def write_metadata_result_file(metadata_result_array, path_to_write):

    _fid = open(path_to_write, 'w')
    # Write header
    _fid.write('sound_event_recording,start_time,end_time,ele,azi,dist\n')
    # Write sources
    for source in metadata_result_array:
        sound_event_recording = source[0] if source[0] is not None else 'None'
        _fid.write('{},'.format(sound_event_recording))
        _fid.write('{},'.format(source[1])) # start_time
        _fid.write('{},'.format(source[2])) # end_time
        _fid.write('{},'.format(source[3])) # ele
        _fid.write('{},'.format(source[4])) # azi
        distance = source[5] if source[5] is not None else 'None'
        _fid.write('{}\n'.format(distance))
    _fid.close()


def build_metadata_result_array_from_event_dict(event_dict, class_id, hop_size, pre_offset, post_offset):


    metadata_result_array = []
    for event_idx, event_values in event_dict.iteritems():
        start_frame = event_values[0][0] - pre_offset
        end_frame = event_values[-1][0] + post_offset

        azis = np.asarray(event_values)[:,1]
        azi = circmedian(np.asarray(azis), unit='deg')
        eles = np.asarray(event_values)[:,2]
        ele = np.median(eles)

        metadata_result_array.append([class_id, start_frame * hop_size, end_frame * hop_size, ele, azi, None])

    # Rebuild metadata_result_array

    return metadata_result_array


def build_result_dict_from_metadata_array(metadata_result_array, hop_size):

    result_dict = {}
    for event_values in metadata_result_array:
        class_idx = event_values[0]
        start_frame = int(np.floor(float(event_values[1]) / hop_size))
        end_frame = int(np.ceil(float(event_values[2]) / hop_size))
        ele = event_values[3]
        azi = event_values[4]

        for frame in range(start_frame, end_frame+1):
            try:
                # result_dict[frame].append([class_idx, azi, ele])
                v = result_dict[frame]
                if len(v) == 3:
                    # Avoid ol3 situations
                    result_dict[frame] = [v, [class_idx, azi, ele]]
            except KeyError:
                # No event at that frame: add new
                result_dict[frame] = [class_idx, azi, ele]

    return result_dict


def assign_metadata_result_classes_from_groundtruth(metadata_file_name, rs_folder_path, gt_folder_path, time_th, dist_th):

    # Load files
    res_list = []
    with open(os.path.join(rs_folder_path, metadata_file_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        res_list = list(reader)[1:]

    gt_list = []
    with open(os.path.join(gt_folder_path, metadata_file_name), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        gt_list = list(reader)[1:]

    # Find the event from gt which is parallel to the one in res
    res_dict = {}
    used_gt_event_idx = []
    for res_e_idx, res_e in enumerate(res_list):
        res_start_time = float(res_e[1])
        res_end_time = float(res_e[2])
        res_ele = float(res_e[3])
        res_azi = float(res_e[4])

        min_diff = 1000
        min_diff_idx = -1

        if len(gt_list) == 0:
            break
        else:
            for gt_e_idx, gt_e in enumerate(gt_list):
                if gt_e_idx not in used_gt_event_idx:
                    gt_start_time = float(gt_e[1])
                    gt_end_time = float(gt_e[2])
                    gt_ele = float(gt_e[3])
                    gt_azi = float(gt_e[4])

                    start_diff = abs(gt_start_time - res_start_time)
                    end_diff = abs(gt_end_time - res_end_time)

                    dist_diff = distance_between_spherical_coordinates_rad(deg2rad(gt_azi),
                                                                           deg2rad(gt_ele),
                                                                           deg2rad(res_azi),
                                                                           deg2rad(res_ele))
                    all_diff = start_diff + end_diff + dist_diff
                    if start_diff <= time_th and end_diff <= time_th and dist_diff <= dist_th:
                        if all_diff < min_diff:
                            min_diff = all_diff
                            min_diff_idx = gt_e_idx

        if min_diff_idx != -1:
            used_gt_event_idx.append(min_diff_idx)
            res_dict[res_e_idx] = min_diff_idx

    output_res_list = copy(res_list)
    last_rs_e_idx = len(res_list)
    for rs_e_idx in range(last_rs_e_idx):
        if rs_e_idx in res_dict.keys():
            gt_e_idx = res_dict[rs_e_idx]
            class_id = gt_list[gt_e_idx][0]
            output_res_list[rs_e_idx][0] = class_id

    write_metadata_result_file(output_res_list, os.path.join(rs_folder_path, metadata_file_name))




