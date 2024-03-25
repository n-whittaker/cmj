# this function reads c3d files
def read_c3d(file, read_mocap=True):
    import os
    import c3d
    import numpy as np
    import pandas as pd
    # check if file exists
    if not os.path.exists(file):
        return {'Error': 'File does not exist'}
    # load c3d into dict
    try:
        try:
            file_id = open(file, 'rb')
            reader = c3d.Reader(file_id)
        except:
            raise ValueError('Reading Error of file')
        # get other info
        info = dict()
        col_of_int = ['DATEOFCAPTURE', 'USER', 'VERSION', 'DESCRIPTION', 'NOTE']
        ssc_info = reader.get('SSCDATAANDPROCESSING')
        if ssc_info is None:
            # if no processing was done at all
            for field in col_of_int:
                info[field] = [None]
        else:
            for field in col_of_int:
                val = ssc_info.get(field)
                if val is None:
                    info[field] = [None]
                elif val.dimensions[0] == 0:
                    info[field] = [None]
                else:
                    try:
                        info[field] = val.string_array
                    except UnicodeDecodeError:
                        if val.bytes_array[0] == b'Zhan\xe9':
                            info[field] = 'Zhane'
                        else:
                            info[field] = val.string_array
        col_of_int = ['BODYMASS', 'HEIGHT']
        ssc_info = reader.get('PROCESSING')
        if ssc_info is None:
            # if no processing was done at all
            for field in col_of_int:
                info[field] = None
        else:
            for field in col_of_int:
                val = ssc_info.get(field)
                if val is None:
                    info[field] = None
                elif val.dimensions[0] == 0:
                    info[field] = None
                else:
                    info[field] = val.float_value
        col_of_int = ['Names']
        ssc_info = reader.get('SUBJECTS')
        if ssc_info is None:
            # if no processing was done at all
            for field in col_of_int:
                info[field] = None
        else:
            for field in col_of_int:
                val = ssc_info.get(field)
                if val is None:
                    info[field] = [None]
                else:
                    info[field] = val.float_value
        # reader.get('TRIAL.CAMERA_RATE').float_value
        col_of_int = ['CAMERA_RATE']
        ssc_info = reader.get('TRIAL')
        for field in col_of_int:
            val = ssc_info.get(field)
            if val is None:
                info[field] = None
            else:
                info[field] = val.float_value
        info['FP_RATE'] = reader.get('ANALOG').get('RATE').float_value
        # get borders of plate 1 and plate 2 to compute mid point of fp to infer
        #   direction. Negative would imply to the right, while positive would imply to
        #   the left
        # get labels
        mocap_labels = reader.point_labels
        mocap_labels = [x.replace(' ', '') for x in mocap_labels]
        mocap_labels = sum([[x + '_x', x + '_y', x + '_z'] for x in mocap_labels], [])
        force_labels = reader.get('ANALOG')
        force_labels = force_labels.get('LABELS')
        force_labels = force_labels.string_array
        force_labels = [x.replace(' ', '') for x in force_labels]
        force_labels = [x.replace('Force.', '') for x in force_labels]
        force_labels = [x.replace('Moment.', '') for x in force_labels]
        if len(force_labels) == 0:
            file_id.close()
            raise ValueError('No ForcePlate Data in C3D please check file')
        fp_border = reader.get('FORCE_PLATFORM')
        fp_border = fp_border.get('CORNERS')
        fp_border = fp_border.float_array
        n_planes, n_points, n_plates = fp_border.shape
        fp_border = np.array(fp_border).reshape(n_plates, n_points, n_planes)
        for count in range(0, n_plates):
            info['xmidposFP' + str(count + 1)] = float(fp_border[count].mean(axis=0)[0])
        frequency_ratio = int(info['FP_RATE'] / info['CAMERA_RATE'])
        if read_mocap:
            frames = reader.header.last_frame - reader.header.first_frame + 1
            mocap_data = np.empty(shape=(frames, len(mocap_labels)))
            force_frames = int(frames * (info['FP_RATE'] / info['CAMERA_RATE']))
            force_data = np.empty(shape=(force_frames, len(force_labels)))
            force_frame, current_frame = [int(x) for x in np.linspace(0, force_frames, num=frames + 1)], 0
            # read in data frame by frame
            for frame, points, analog in reader.read_frames(copy=False):
                if frame < reader.header.first_frame or frame > reader.header.last_frame:
                    continue
                mocap_data[current_frame, :] = np.array(points[:, 0:3]).reshape(1, len(mocap_labels))
                force_data[force_frame[current_frame]:force_frame[current_frame + 1], :] = \
                    np.array(analog).reshape(frequency_ratio, len(force_labels))
                current_frame += 1
            # define MoCap Data
            frame_rate = (1 / info['CAMERA_RATE'])
            n_frames = reader.last_frame() - reader.first_frame() + 1
            frames = np.linspace(reader.first_frame(), reader.last_frame(), num=n_frames)
            frames = (frames - 1) * frame_rate
            mocap_data = pd.DataFrame(mocap_data, columns=mocap_labels, index=frames)
            # define Force Data
            ratio = int(reader.analog_rate / reader.point_rate)
            frames = np.linspace(reader.first_frame(), reader.last_frame() + 1 - (1 / ratio),
                                 num=n_frames * ratio)
            frames = (frames * 10).round() / 10
            frames = (frames - 1) * frame_rate
            force_data = pd.DataFrame(force_data, columns=force_labels, index=frames)
            # invert the plates like Vicon would do
            labels, label_weights = [], []
            for col, w in [['Fx1', -1], ['Fy1', -1], ['Fz1', -1], ['Fx2', -1], ['Fy2', -1], ['Fz2', -1]]:
                if col in force_data:
                    labels += [col]
                    label_weights += [w]
            force_data = force_data[labels] * label_weights
        else:
            mocap_data = []
            force_data = []
        file_id.close()
        return {'MoCap': mocap_data, 'GRF': force_data, 'Info': info}
    except OSError as err:
        hu = 5
        return {'Error': '{}'.format(err)}
    except ValueError as err:
        return {'Error': '{}'.format(err)}
    except:
        hu = 5
        return {'Error': 'Previously undiscovered error'}
