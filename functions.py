#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import mne
from mne.io import concatenate_raws

def seed_shuffle(input, seed=1):
    np.random.seed(seed)
    np.random.shuffle(input)
    return input

def empty_append_row(a, b):
    if type(a) is np.ndarray and type(b) is np.ndarray:
        return np.append(a, b, axis=0)
    return a or b

def get_act_data(input_data, sub_range):
    output = None
    for i in sub_range:
        output = empty_append_row(output, input_data[(input_data[:,-1]==i)])
    return output

def get_act_time_sequences(input_data, window_size, step):

    output_X = []
    output_y = []
    output_u = []
    start = 0
    y = input_data[start][-2]
    u = input_data[start][-1]
    while (start + window_size - 1) < (input_data.shape[0]):
        end = start + window_size - 1

        if input_data[end][-2] != y or input_data[end][-1] != u:
            y = input_data[end][-2]
            u = input_data[end][-1]
            for i in range(end, 0, -1):
                if input_data[i][-2] != y or input_data[i][-1] != u:
                    start = i + 1
                    # print('start',start)
                    break
        else:
            output_X.append(input_data[start: end + 1, :-2])
            output_y.append(y)
            output_u.append(u)
            start = start + step
    return np.array(output_X), np.array(output_y), np.array(output_u)

def get_eeg_data(event_codes, sub_list):

    """
    get EEG data
    sub 88, 89, 92, 100 have wrong data
    event_codes = [4, 8, 12]  # imagine opening and closing left or right fist
    event_codes = [3, 7, 11]  # really opening and closing left or right fist
    shape: [#trial:64:time_length]
    see
    https://www.physionet.org/pn4/eegmmidb/
    """

    physionet_paths = [mne.datasets.eegbci.load_data(sub_id, event_codes) for sub_id in sub_list]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto')
             for path in physionet_paths]

    raw = concatenate_raws(parts)

    # add filter
    # raw.filter(4., 30., fir_design='firwin', skip_by_annotation='edge')

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    epoched = mne.Epochs(raw, events, dict(left=2, right=3), tmin=1, tmax=4.1, proj=False, picks=picks,
                         baseline=None, preload=True)
    X = (epoched.get_data() * 1e6).astype(np.float32)  #fetures, shape: [#trial:64:time_length]
    y = (epoched.events[:, 2] - 2).astype(int)  #label: rest:1, left: 2, right:3, minus two to get labels 0, 1

    return X, y



"""
Time Window
"""


def windows(data, size, step):

    start = 0
    while (start + size) <= data.shape[0]:
        yield int(start), int(start + size)
        start += step


def segment_signal_without_transition(data, window_size, step):

    segments = []
    for (start, end) in windows(data, window_size, step):
        if len(data[start:end]) == window_size:
            segments = segments + [data[start:end]]
    return np.array(segments)


def segment_dataset(X, window_size, step):

    win_x = []
    for i in range(X.shape[0]):
        win_x = win_x + [segment_signal_without_transition(X[i],
                                                           window_size, step)]
    win_x = np.array(win_x)
    return win_x


"""
Neural Networks
"""


def weight_variable(shape, name = None):

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name = None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


"""
Convolutional Neural Networks
"""


def conv1d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv1d(x, W, stride=kernel_stride, padding="VALID")


def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding="VALID")


def conv3d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv3d(x, W, strides=[1, kernel_stride, kernel_stride, kernel_stride, 1], padding="VALID")


def apply_conv1d(x, filter_width, in_channels, out_channels, kernel_stride, train_phase):
    weight = weight_variable([filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    conv_1d = tf.add(conv1d(x, weight, kernel_stride), bias)
    conv_1d_bn = batch_norm_cnv_1d(conv_1d, train_phase)
    return tf.nn.elu(conv_1d_bn)


def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride, train_phase):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    conv_2d = tf.add(conv2d(x, weight, kernel_stride), bias)
    conv_2d_bn = batch_norm_cnv_2d(conv_2d, train_phase)
    return tf.nn.elu(conv_2d_bn)


def apply_conv3d(x, filter_depth, filter_height, filter_width, in_channels, out_channels, kernel_stride, train_phase):
    weight = weight_variable([filter_depth, filter_height, filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    conv_3d = tf.add(conv3d(x, weight, kernel_stride), bias)
    conv_3d_bn = batch_norm_cnv_3d(conv_3d, train_phase)
    return tf.nn.elu(conv_3d_bn)


def batch_norm_cnv_3d(inputs, train_phase):
    return tf.layers.batch_normalization(inputs, axis=4, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


def batch_norm_cnv_2d(inputs, train_phase):
    return tf.layers.batch_normalization(inputs, axis=3, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


def batch_norm_cnv_1d(inputs, train_phase):
    return tf.layers.batch_normalization(inputs, axis=2, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


def batch_norm(inputs, train_phase):
    return tf.layers.batch_normalization(inputs, axis=1, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding="VALID")


def apply_max_pooling3d(x, pooling_depth, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool3d(x, ksize=[1, pooling_depth, pooling_height, pooling_width, 1],
                            strides=[1, pooling_stride, pooling_stride, pooling_stride, 1], padding="VALID")


def apply_fully_connect(x, x_size, fc_size, train_phase):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    fc = tf.add(tf.matmul(x, fc_weight), fc_bias)
    fc_bn = batch_norm(fc, train_phase)
    return tf.nn.elu(fc_bn)


def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    return tf.add(tf.matmul(x, readout_weight), readout_bias)
