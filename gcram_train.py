import scipy.io as sc
import numpy as np
from functions import *
from act_models import *
from gram_model import *
# from gcram_model import *
from grcam_model import *

path = 'Data/'

# dict: [filename, sub, class]
datasets_dict = {'MH8': ['MHEALTH_8class', 10, 8],
                 'MH6': ['MHEALTH_balance', 10, 6],  # better
                 # 'OPP': ['Opp_Gesture', 4, 17],
                 # 'OPP5': ['Opp_loco_5', 4, 5],
                 'PMP8': ['PAMAP2_8class', 8, 8],
                 'PMP6': ['PAMAP2_Protocol_6_classes_balance', 8, 6],  # better
                 'UCI': ['UCI_train_raw', 10, 6],
                 # 'EEG': ['EEG_10sub_4act', 10, 4],
                 'MARS': ['MARS', 8, 5]}


# get data
dataset = 'MH6'
target_id = 3
source_size = 8000
target_size = 2000
batch_size = 5000
training_epochs = 2000

# data process
# window 20, step 10 is best for MHEALTH
window_size = 20
step = 10

# training
learning_rate = 5e-3
learning_rate_decay_factor = 0.97
min_learning_rate = 1e-5
max_gradient_norm = 5.0

# model
g_size = 128  # Size of theta_g^0
l_size = 128  # Size of theta_g^1
glimpse_output_size = 220  # Output size of Glimpse Network
cell_size = 220  # Size of LSTM cell
nb_glimpses = 30  # Number of glimpses: 30
variance = 0.22  # Gaussian variance for Location Network
# M = 1 # Monte Carlo sampling, see Eq(2) (not used)
glimpse_time_down_scale = 4
glimpse_location_down_scale = 4

file_name = datasets_dict[dataset][0]
file_data = sc.loadmat(path + file_name + '.mat')
file_data = file_data[file_name]

nb_subjects = datasets_dict[dataset][1]
nb_classes = datasets_dict[dataset][2]

source_range = list(range(nb_subjects))
source_range.remove(target_id)
target_range = [target_id]

source_data = get_act_data(file_data, source_range)
target_data = get_act_data(file_data, target_range)

if dataset == 'UCI':
    print('dataset is UCI')
    source_data = sc.loadmat(path + 'UCI_train_raw' + '.mat')
    source_data = source_data['UCI_train_raw']

    target_data = sc.loadmat(path + 'UCI_test_raw' + '.mat')
    target_data = target_data['UCI_test_raw']

# print(source_data.shape)

source_X, source_y, source_u = get_act_time_sequences(source_data, window_size, step)
target_X, target_y, target_u = get_act_time_sequences(target_data, window_size, step)

nb_feature = source_X.shape[-1]
# source_X = source_X.reshape([source_X.shape[0], -1])
# target_X = target_X.reshape([target_X.shape[0], -1])

source_X, source_y, source_u = seed_shuffle(source_X, 1), seed_shuffle(source_y, 1), seed_shuffle(source_u, 1)
target_X, target_y, target_u = seed_shuffle(target_X, 1), seed_shuffle(target_y, 1), seed_shuffle(target_u, 1)

source_X, source_y, source_u = source_X[: source_size], source_y[: source_size], source_u[: source_size]
target_X, target_y, target_u = target_X[: target_size], target_y[: target_size], target_u[: target_size]


# model
img_height = window_size
img_width = nb_feature

glimpse_width = max(img_width // glimpse_location_down_scale, 1)
glimpse_height = max(img_height // glimpse_time_down_scale, 1)


ram = GlobalRecurrentConvolutionalAttentionModel(img_width=img_width,
                                    img_height=img_height,
                                    nb_locations=3,
                                    glimpse_width=glimpse_width,
                                    glimpse_height=glimpse_height,
                                    g_size=g_size,
                                    l_size=l_size,
                                    glimpse_output_size=glimpse_output_size,
                                    loc_dim=1,   # (x,y)
                                    time_dim=1,
                                    variance=variance,
                                    cell_size=cell_size,
                                    nb_glimpses=nb_glimpses,
                                    nb_classes=nb_classes,
                                    learning_rate=learning_rate,
                                    learning_rate_decay_factor=learning_rate_decay_factor,
                                    min_learning_rate=min_learning_rate,
                                    nb_training_batch=source_X.shape[0]//batch_size,
                                    max_gradient_norm=max_gradient_norm,
                                    is_training=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())

    best_acc_per_run = 0

    for epoch in range(training_epochs):

        # training process
        for b in range(source_X.shape[0] // batch_size):
            source_batch_x = source_X[batch_size * b: batch_size * (b + 1)]
            source_batch_y = source_y[batch_size * b: batch_size * (b + 1)]

            # c = session.run(
            #     ram.init_glimpse,
            #     feed_dict={ram.img_ph: source_batch_x,
            #                ram.lbl_ph: source_batch_y,
            #                })
            # print('c', c.shape)
            #
            # a = session.run(
            #     ram.init_glimpse_cooperate,
            #     feed_dict={ram.img_ph: source_batch_x,
            #                ram.lbl_ph: source_batch_y,
            #                })
            # print(a.shape)

            # simgs_ph, simgs_ph_re, sh_fc1, sconv_2d_1st, sconv_2d_2nd, sconv_2d_flat = session.run(
            #     [ram.imgs_ph, ram.imgs_ph_re, ram.h_fc1, ram.conv_2d_1st, ram.conv_2d_2nd, ram.conv_2d_flat],
            #     feed_dict={ram.img_ph: source_batch_x,
            #                ram.lbl_ph: source_batch_y,
            #                })
            # print('\n\n\n cnn shape:\n', simgs_ph.shape, simgs_ph_re.shape, sh_fc1.shape, sconv_2d_1st.shape, sconv_2d_2nd.shape, sconv_2d_flat.shape)

            _, loss_source_y, accuracy_source_y = session.run(
                [ram.train_op, ram.cross_entropy, ram.accuracy],
                feed_dict={ram.img_ph: source_batch_x,
                           ram.lbl_ph: source_batch_y,
                           })

        # test

        loss_target_y, accuracy_target_y, prediction = session.run(
            [ram.cross_entropy, ram.accuracy, ram.pred],
            feed_dict={ram.img_ph: target_X,
                       ram.lbl_ph: target_y,
                       })

        # confusion_matrix
        # confusion_matrix = [[0] * nb_classes for _ in range(nb_classes)]
        # for i in range(target_X.shape[0]):
        #     confusion_matrix[int(target_y[i])][int(prediction[i])] += 1
        # for i in range(nb_classes):
        #     confusion_matrix[i] = [100 * j / sum(confusion_matrix[i]) for j in confusion_matrix[i]]
        # print('confusion_matrix:')
        # for i in range(len(confusion_matrix)):
        #     print(confusion_matrix[i])

        if accuracy_target_y > best_acc_per_run:
            best_acc_per_run = accuracy_target_y

        print(epoch, dataset, 'target_id: ', target_id,
              'loss: ', loss_source_y, loss_target_y,
              'acc: ', accuracy_source_y, accuracy_target_y,
              best_acc_per_run)
