## FILE UTILS
import numpy as np



def write_output_result_file(output_result_dict, path_to_write):
    """
    (Adapted from metrics.evaluation_metrics.write_output_format_file)
    :param output_result_dict:
    :param path_to_write:
    :return:
    """

    # Ensure ascending order
    frames = output_result_dict.keys()
    frames.sort()

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




def build_result_dict_from_metadata_array(metadata_result_array, hop_size):

    result_dict = {}
    for event_values in metadata_result_array:
        class_idx = event_values[0]
        start_frame = int(np.floor(float(event_values[1]) / hop_size))
        end_frame = int(np.ceil(float(event_values[2]) / hop_size))
        ele = event_values[3]
        azi = event_values[4]

        for frame in range(start_frame, end_frame):
            try:
                # result_dict[frame].append([class_idx, azi, ele])
                v = result_dict[frame]
                result_dict[frame] = [v, [class_idx, azi, ele]]
            except KeyError:
                result_dict[frame] = [class_idx, azi, ele]

    return result_dict

