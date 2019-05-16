## FILE UTILS



def write_output_result_file(output_result_dict, path_to_write):
    """
    (Adapted from metrics.evaluation_metrics.write_output_format_file)
    :param output_result_dict:
    :param path_to_write:
    :return:
    """

    _fid = open(path_to_write, 'w')
    for _frame_ind in output_result_dict.keys():
        _value = output_result_dict[_frame_ind]
        if len(_value) == 3:
            # this is only one point
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
        elif len(_value) == 2:
            # two sources
            for _v in _value:
                _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_v[0]), int(_v[1]), int(_v[2])))
    _fid.close()
#
#
# def convert_output_result_dict_2_metadata_result_array(dict, hop_size=0.02):
#
#
#     # Dictionary keys are ordered in time
#




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


