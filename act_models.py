from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io as sc
from functions import *
from flip_gradient import flip_gradient


class CNNCrossSub_act(object):
    def __init__(self, input_height, input_width, input_channel_num,
                 num_labels_y,
                 pooling_height_1st, pooling_width_1st, pooling_stride_1st,
                 learning_rate, lambda_weight):

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel_num = input_channel_num
        self.num_labels_y = num_labels_y
        # pooling
        self.pooling_height_1st = pooling_height_1st # 1
        self.pooling_width_1st = pooling_width_1st  # no window: 75, with window: 15
        self.pooling_stride_1st = pooling_stride_1st  # no window: 10, with window:  3

        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight

        self._build_model()

    def _build_model(self):

        """
        CNN structure
        """

        # kernel parameter
        kernel_height_1st = self.input_height // 10
        kernel_width_1st = self.input_width
        kernel_stride = 1
        conv_channel_num = 40

        # set L2 penalty
        lambda_l2 = 0.0005

        self.X = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width], name='X')
        X_image = tf.reshape(self.X, [-1, self.input_height, self.input_width, self.input_channel_num])
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')
        self.u = tf.placeholder(tf.int64, shape=[None], name='u')
        # self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        with tf.variable_scope('feature_extractor'):

            # first CNN layer
            c_w_1 = weight_variable([kernel_height_1st, kernel_width_1st,
                                            self.input_channel_num, conv_channel_num],
                                           'c_w_1')
            c_b_1 = bias_variable([conv_channel_num], 'c_b_1')
            conv_2d = tf.add(conv2d(X_image, c_w_1, kernel_stride), c_b_1)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            conv_1 = tf.nn.elu(conv_2d)

            pool_1 = apply_max_pooling(conv_1, self.pooling_height_1st, self.pooling_width_1st, self.pooling_stride_1st)

            # flattern the last layer of cnn
            shape = pool_1.get_shape().as_list()
            pool_1_flat = tf.reshape(pool_1, [-1, shape[1] * shape[2] * shape[3]])
            self.feature = pool_1_flat

        with tf.variable_scope('y_predictor'):

            fc_drop_y = tf.nn.dropout(pool_1_flat, self.keep_prob)

            y_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_y], 'y_w_1')
            y_b_1 = bias_variable([self.num_labels_y], 'y_b_1')
            logits_y = tf.add(tf.matmul(fc_drop_y, y_w_1), y_b_1)

            softmax_y = tf.nn.softmax(logits_y)
            pred_y = tf.argmax(softmax_y, 1)
            self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(pred_y, self.y), tf.float32))

            l2 = lambda_l2 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss_y = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=self.y)) + l2

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_y)

class DNN(object):
    def __init__(self, input_height, input_width, input_channel_num,
                 num_labels_y,
                 pooling_height_1st, pooling_width_1st, pooling_stride_1st,
                 learning_rate, lambda_weight):

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel_num = input_channel_num
        self.num_labels_y = num_labels_y
        # pooling
        self.pooling_height_1st = pooling_height_1st # 1
        self.pooling_width_1st = pooling_width_1st  # no window: 75, with window: 15
        self.pooling_stride_1st = pooling_stride_1st  # no window: 10, with window:  3

        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight

        self._build_model()

    def _build_model(self):

        """
        CNN structure
        """

        # set L2 penalty
        lambda_l2 = 0.0005

        self.X = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width], name='X')
        X_image = tf.reshape(self.X, [-1, self.input_height * self.input_width])
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')
        self.u = tf.placeholder(tf.int64, shape=[None], name='u')
        # self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        with tf.variable_scope('feature_extractor'):

            # first CNN layer
            c_w_1 = weight_variable([self.input_height * self.input_width, self.input_height * self.input_width//2],
                                           'c_w_1')
            c_b_1 = bias_variable([self.input_height * self.input_width//2], 'c_b_1')
            conv_2d = tf.add(tf.matmul(X_image, c_w_1), c_b_1)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            conv_1 = tf.nn.elu(conv_2d)
            self.feature = conv_1

        with tf.variable_scope('y_predictor'):

            fc_drop_y = tf.nn.dropout(self.feature, self.keep_prob)

            y_w_1 = weight_variable([self.input_height * self.input_width//2, self.num_labels_y], 'y_w_1')
            y_b_1 = bias_variable([self.num_labels_y], 'y_b_1')
            logits_y = tf.add(tf.matmul(fc_drop_y, y_w_1), y_b_1)

            softmax_y = tf.nn.softmax(logits_y)
            pred_y = tf.argmax(softmax_y, 1)
            self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(pred_y, self.y), tf.float32))

            l2 = lambda_l2 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss_y = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=self.y)) + l2

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_y)


class CNNAdversarial_act(object):
    def __init__(self, input_height, input_width, input_channel_num,
                 num_labels_y, num_labels_u,
                 pooling_height_1st, pooling_width_1st, pooling_stride_1st,
                 learning_rate, lambda_weight):

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel_num = input_channel_num
        self.num_labels_y = num_labels_y
        self.num_labels_u = num_labels_u
        # pooling
        self.pooling_height_1st = pooling_height_1st # 1
        self.pooling_width_1st = pooling_width_1st  # no window: 75, with window: 15
        self.pooling_stride_1st = pooling_stride_1st  # no window: 10, with window:  3

        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight

        self._build_model()

    def _build_model(self):

        """
        CNN structure
        """

        # kernel parameter
        kernel_height_1st = self.input_height // 10
        kernel_width_1st = self.input_width
        kernel_stride = 1
        conv_channel_num = 40

        # set L2 penalty
        lambda_l2 = 0.0005

        self.X = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width], name='X')
        X_image = tf.reshape(self.X, [-1, self.input_height, self.input_width, self.input_channel_num])
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')
        self.u = tf.placeholder(tf.int64, shape=[None], name='u')
        # self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        with tf.variable_scope('feature_extractor'):

            # first CNN layer
            c_w_1 = weight_variable([kernel_height_1st, kernel_width_1st,
                                            self.input_channel_num, conv_channel_num],
                                           'c_w_1')
            c_b_1 = bias_variable([conv_channel_num], 'c_b_1')
            conv_2d = tf.add(conv2d(X_image, c_w_1, kernel_stride), c_b_1)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            conv_1 = tf.nn.elu(conv_2d)

            pool_1 = apply_max_pooling(conv_1, self.pooling_height_1st, self.pooling_width_1st, self.pooling_stride_1st)

            # flattern the last layer of cnn
            shape = pool_1.get_shape().as_list()
            pool_1_flat = tf.reshape(pool_1, [-1, shape[1] * shape[2] * shape[3]])
            self.feature = pool_1_flat

        with tf.variable_scope('decoder'):

            d_w_1 = weight_variable([shape[1] * shape[2] * shape[3],
                                            self.input_height * self.input_width * self.input_channel_num],
                                           'd_w_1')

            d_b_1 = bias_variable([self.input_height * self.input_width * self.input_channel_num], 'd_b_1')
            decoded = tf.add(tf.matmul(self.feature, d_w_1), d_b_1)
            reconstruction = tf.reshape(decoded, [-1, self.input_height, self.input_width, self.input_channel_num])

            self.loss_ae = tf.losses.mean_squared_error(labels=X_image, predictions=reconstruction)
            self.optimizer_ae = \
                tf.train.AdamOptimizer(20 * self.learning_rate).minimize(self.loss_ae,
                                                                         var_list=[c_w_1, c_b_1, d_w_1, d_b_1])

        with tf.variable_scope('y_predictor'):

            fc_drop_y = tf.nn.dropout(pool_1_flat, self.keep_prob)

            y_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_y], 'y_w_1')
            y_b_1 = bias_variable([self.num_labels_y], 'y_b_1')
            logits_y = tf.add(tf.matmul(fc_drop_y, y_w_1), y_b_1)

            softmax_y = tf.nn.softmax(logits_y)
            pred_y = tf.argmax(softmax_y, 1)
            self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(pred_y, self.y), tf.float32))

            l2 = lambda_l2 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss_y = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=self.y)) + l2

            self.optimizer_fy_y = \
                tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_y,
                                                                    var_list=[c_w_1, c_b_1, y_w_1, y_b_1])

        with tf.variable_scope('u_predictor_no_flip'):  # in order to train user classifier

            u_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_u], 'u_w_1')
            u_b_1 = bias_variable([self.num_labels_u], 'u_b_1')

            feat_no_flip = self.feature
            # feat_no_flip = flip_gradient(self.feature, self.lambda_weight)
            fc_drop_u_no_flip = tf.nn.dropout(feat_no_flip, self.keep_prob)

            logits_u_no_flip = tf.add(tf.matmul(fc_drop_u_no_flip, u_w_1), u_b_1)

            softmax_u_no_flip = tf.nn.softmax(logits_u_no_flip)
            pred_u_no_flip = tf.argmax(softmax_u_no_flip, 1)
            self.accuracy_u_no_flip = tf.reduce_mean(tf.cast(tf.equal(pred_u_no_flip, self.u), tf.float32))

            self.loss_u_no_flip = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_u_no_flip, labels=self.u)) + l2

            self.optimizer_u_no_flip = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_u_no_flip,
                                                                                               var_list=[u_w_1, u_b_1])

        with tf.variable_scope('u_predictor_with_flip'):  # in order to train feature extractor

            feat = flip_gradient(self.feature, self.lambda_weight)
            fc_drop_u = tf.nn.dropout(feat, self.keep_prob)
            logits_u = tf.add(tf.matmul(fc_drop_u, u_w_1), u_b_1)

            softmax_u = tf.nn.softmax(logits_u)
            pred_u = tf.argmax(softmax_u, 1)
            self.accuracy_u = tf.reduce_mean(tf.cast(tf.equal(pred_u, self.u), tf.float32))

            # train feature extractor parameters, maximize user classifier loss
            self.loss_u_flip = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_u, labels=self.u)) + l2

            self.optimizer_f_uflip = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_u_flip,
                                                                                         var_list=[c_w_1, c_b_1])

            # train feature extractor and decoder parameters, maximize user classifier loss and minimize AE loss
            loss_ae_cnn = self.lambda_weight * self.loss_ae + self.loss_u_flip
            self.optimizer_ae_cnn = \
                tf.train.AdamOptimizer(self.learning_rate).minimize(loss_ae_cnn,
                                                                    var_list=[c_w_1, c_b_1, d_w_1, d_b_1])


class CNNAdversarial_act_cfg(object):
    def __init__(self, input_height, input_width, input_channel_num,
                 num_labels_y, num_labels_u,
                 pooling_height_1st, pooling_width_1st, pooling_stride_1st,
                 learning_rate, lambda_weight):

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel_num = input_channel_num
        self.num_labels_y = num_labels_y
        self.num_labels_u = num_labels_u
        # pooling
        self.pooling_height_1st = pooling_height_1st # 1
        self.pooling_width_1st = pooling_width_1st  # no window: 75, with window: 15
        self.pooling_stride_1st = pooling_stride_1st  # no window: 10, with window:  3

        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight

        self._build_model()

    def _build_model(self):

        """
        CNN structure
        """

        # kernel parameter
        kernel_height_1st = self.input_height // 10
        kernel_width_1st = self.input_width
        kernel_stride = 1
        conv_channel_num = 40

        # set L2 penalty
        lambda_l2 = 0.0005

        self.X = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width], name='X')
        X_image = tf.reshape(self.X, [-1, self.input_height, self.input_width, self.input_channel_num])
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')
        self.u = tf.placeholder(tf.int64, shape=[None], name='u')
        # self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        with tf.variable_scope('feature_extractor'):

            # first CNN layer
            c_w_1 = weight_variable([kernel_height_1st, kernel_width_1st,
                                            self.input_channel_num, conv_channel_num],
                                           'c_w_1')
            c_b_1 = bias_variable([conv_channel_num], 'c_b_1')
            conv_2d = tf.add(conv2d(X_image, c_w_1, kernel_stride), c_b_1)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            conv_1 = tf.nn.elu(conv_2d)

            pool_1 = apply_max_pooling(conv_1, self.pooling_height_1st, self.pooling_width_1st, self.pooling_stride_1st)

            # flattern the last layer of cnn
            shape = pool_1.get_shape().as_list()
            pool_1_flat = tf.reshape(pool_1, [-1, shape[1] * shape[2] * shape[3]])
            self.feature = pool_1_flat

        with tf.variable_scope('decoder'):

            d_w_1 = weight_variable([shape[1] * shape[2] * shape[3],
                                            self.input_height * self.input_width * self.input_channel_num],
                                           'd_w_1')

            d_b_1 = bias_variable([self.input_height * self.input_width * self.input_channel_num], 'd_b_1')
            decoded = tf.add(tf.matmul(self.feature, d_w_1), d_b_1)
            reconstruction = tf.reshape(decoded, [-1, self.input_height, self.input_width, self.input_channel_num])

            self.loss_ae = tf.losses.mean_squared_error(labels=X_image, predictions=reconstruction)
            self.optimizer_ae = \
                tf.train.AdamOptimizer(20 * self.learning_rate).minimize(self.loss_ae,
                                                                         var_list=[c_w_1, c_b_1, d_w_1, d_b_1])

        with tf.variable_scope('y_predictor'):

            fc_drop_y = tf.nn.dropout(pool_1_flat, self.keep_prob)

            y_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_y], 'y_w_1')
            y_b_1 = bias_variable([self.num_labels_y], 'y_b_1')
            logits_y = tf.add(tf.matmul(fc_drop_y, y_w_1), y_b_1)

            softmax_y = tf.nn.softmax(logits_y)
            pred_y = tf.argmax(softmax_y, 1)
            self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(pred_y, self.y), tf.float32))

            l2 = lambda_l2 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss_y = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=self.y)) + l2

            self.optimizer_fy_y = \
                tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_y,
                                                                    var_list=[c_w_1, c_b_1, y_w_1, y_b_1])

        with tf.variable_scope('u_predictor_no_flip'):  # in order to train user classifier

            u_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_u], 'u_w_1')
            u_b_1 = bias_variable([self.num_labels_u], 'u_b_1')

            feat_no_flip = self.feature
            # feat_no_flip = flip_gradient(self.feature, self.lambda_weight)
            fc_drop_u_no_flip = tf.nn.dropout(feat_no_flip, self.keep_prob)

            logits_u_no_flip = tf.add(tf.matmul(fc_drop_u_no_flip, u_w_1), u_b_1)

            softmax_u_no_flip = tf.nn.softmax(logits_u_no_flip)
            pred_u_no_flip = tf.argmax(softmax_u_no_flip, 1)
            self.accuracy_u_no_flip = tf.reduce_mean(tf.cast(tf.equal(pred_u_no_flip, self.u), tf.float32))

            self.loss_u_no_flip = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_u_no_flip, labels=self.u)) + l2

            self.optimizer_u_no_flip = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_u_no_flip,
                                                                                               var_list=[u_w_1, u_b_1])
            cfg = tf.gradients(self.loss_u_no_flip, [u_w_1, u_b_1])[0]

        with tf.variable_scope('u_predictor_with_flip'):  # in order to train feature extractor

            self.feature = tf.add(self.feature, cfg)
            feat = flip_gradient(self.feature, self.lambda_weight)
            fc_drop_u = tf.nn.dropout(feat, self.keep_prob)
            logits_u = tf.add(tf.matmul(fc_drop_u, u_w_1), u_b_1)

            softmax_u = tf.nn.softmax(logits_u)
            pred_u = tf.argmax(softmax_u, 1)
            self.accuracy_u = tf.reduce_mean(tf.cast(tf.equal(pred_u, self.u), tf.float32))

            # train feature extractor parameters, maximize user classifier loss
            self.loss_u_flip = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_u, labels=self.u)) + l2

            self.optimizer_f_uflip = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_u_flip,
                                                                                         var_list=[c_w_1, c_b_1])

            # train feature extractor and decoder parameters, maximize user classifier loss and minimize AE loss
            loss_ae_cnn = self.lambda_weight * self.loss_ae + self.loss_u_flip
            self.optimizer_ae_cnn = \
                tf.train.AdamOptimizer(self.learning_rate).minimize(loss_ae_cnn,
                                                                    var_list=[c_w_1, c_b_1, d_w_1, d_b_1])

class StarAct(object):
    def __init__(self, input_height, input_width, input_channel_num,
                 num_labels_y, num_labels_u,
                 pooling_height_1st, pooling_width_1st, pooling_stride_1st,
                 learning_rate, lambda_weight):

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel_num = input_channel_num
        self.num_labels_y = num_labels_y
        self.num_labels_u = num_labels_u
        # pooling
        self.pooling_height_1st = pooling_height_1st # 1
        self.pooling_width_1st = pooling_width_1st  # no window: 75, with window: 15
        self.pooling_stride_1st = pooling_stride_1st  # no window: 10, with window:  3

        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight

        self._build_model()

    def _build_model(self):

        """
        CNN structure
        """

        # kernel parameter
        kernel_height_1st = self.input_height // 10
        kernel_width_1st = self.input_width
        kernel_stride = 1
        conv_channel_num = 40

        # set L2 penalty
        lambda_l2 = 0.0005

        self.X1 = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width], name='X1')
        X1_image = tf.reshape(self.X1, [-1, self.input_height, self.input_width, self.input_channel_num])
        self.X2 = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width], name='X2')
        X2_image = tf.reshape(self.X2, [-1, self.input_height, self.input_width, self.input_channel_num])
        self.y1 = tf.placeholder(tf.int64, shape=[None], name='y1')
        self.y2 = tf.placeholder(tf.int64, shape=[None], name='y2')
        self.u1 = tf.placeholder(tf.int64, shape=[None], name='u1')
        self.u2 = tf.placeholder(tf.int64, shape=[None], name='u2')
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        with tf.variable_scope('encoder1'):

            # first CNN layer
            en1_c_w_1 = weight_variable([kernel_height_1st, kernel_width_1st,
                                     self.input_channel_num, conv_channel_num],
                                    'en1_c_w_1')
            en1_c_b_1 = bias_variable([conv_channel_num], 'en1_c_b_1')
            en1_conv_2d = tf.add(conv2d(X1_image, en1_c_w_1, kernel_stride), en1_c_b_1)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            en1_conv_1 = tf.nn.elu(en1_conv_2d)

            en1_pool_1 = apply_max_pooling(en1_conv_1, self.pooling_height_1st, self.pooling_width_1st, self.pooling_stride_1st)

            # flattern the last layer of cnn
            shape = en1_pool_1.get_shape().as_list()
            en1_pool_1_flat = tf.reshape(en1_pool_1, [-1, shape[1] * shape[2] * shape[3]])

            share_w_1 = weight_variable([shape[1] * shape[2] * shape[3], shape[1] * shape[2] * shape[3]],
                                        'share_w_1')
            share_b_1 = bias_variable([shape[1] * shape[2] * shape[3]], 'share_b_1')
            self.en1_feature = tf.add(tf.matmul(en1_pool_1_flat, share_w_1), share_b_1)

        with tf.variable_scope('encoder2'):

            # first CNN layer
            en2_c_w_1 = weight_variable([kernel_height_1st, kernel_width_1st,
                                     self.input_channel_num, conv_channel_num],
                                    'en2_c_w_1')
            en2_c_b_1 = bias_variable([conv_channel_num], 'en2_c_b_1')
            en2_conv_2d = tf.add(conv2d(X2_image, en2_c_w_1, kernel_stride), en2_c_b_1)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            en2_conv_1 = tf.nn.elu(en2_conv_2d)

            en2_pool_1 = apply_max_pooling(en2_conv_1, self.pooling_height_1st, self.pooling_width_1st, self.pooling_stride_1st)

            # flattern the last layer of cnn
            shape = en2_pool_1.get_shape().as_list()
            en2_pool_1_flat = tf.reshape(en2_pool_1, [-1, shape[1] * shape[2] * shape[3]])
            self.en2_feature = tf.add(tf.matmul(en2_pool_1_flat, share_w_1), share_b_1)

        with tf.variable_scope('decoder1'):

            de1_d_w_1 = weight_variable([shape[1] * shape[2] * shape[3],
                                            self.input_height * self.input_width * self.input_channel_num],
                                           'de1_d_w_1')

            de1_d_b_1 = bias_variable([self.input_height * self.input_width * self.input_channel_num], 'de1_d_b_1')
            de1_decoded = tf.add(tf.matmul(self.en1_feature, de1_d_w_1), de1_d_b_1)
            de1_reconstruction = tf.reshape(de1_decoded, [-1, self.input_height, self.input_width, self.input_channel_num])

            self.loss_ae1 = tf.losses.mean_squared_error(labels=X1_image, predictions=de1_reconstruction)

        with tf.variable_scope('decoder2'):

            de2_d_w_1 = weight_variable([shape[1] * shape[2] * shape[3],
                                            self.input_height * self.input_width * self.input_channel_num],
                                           'de2_d_w_1')

            de2_d_b_1 = bias_variable([self.input_height * self.input_width * self.input_channel_num], 'de2_d_b_1')
            de2_decoded = tf.add(tf.matmul(self.en2_feature, de2_d_w_1), de2_d_b_1)
            de2_reconstruction = tf.reshape(de2_decoded, [-1, self.input_height, self.input_width, self.input_channel_num])

            self.loss_ae2 = tf.losses.mean_squared_error(labels=X2_image, predictions=de2_reconstruction)

        with tf.variable_scope('cycle1'):

            # encoder1
            cycle1_encoded1 = self.en1_feature

            # decoder2
            cycle1_decoded = tf.add(tf.matmul(cycle1_encoded1, de2_d_w_1), de2_d_b_1)
            cycle1_reconstruction = tf.reshape(cycle1_decoded, [-1, self.input_height, self.input_width, self.input_channel_num])

            # encoder2
            cycle1_en2_conv_2d = tf.add(conv2d(cycle1_reconstruction, en2_c_w_1, kernel_stride), en2_c_b_1)
            cycle1_en2_conv_1 = tf.nn.elu(cycle1_en2_conv_2d)
            cycle1_en2_pool_1 = apply_max_pooling(cycle1_en2_conv_1, self.pooling_height_1st, self.pooling_width_1st,
                                           self.pooling_stride_1st)
            shape = cycle1_en2_pool_1.get_shape().as_list()
            cycle1_en2_pool_1_flat = tf.reshape(cycle1_en2_pool_1, [-1, shape[1] * shape[2] * shape[3]])
            cycle1_encoded2 = tf.add(tf.matmul(cycle1_en2_pool_1_flat, share_w_1), share_b_1)

            self.loss_cycle1 = tf.losses.mean_squared_error(labels=cycle1_encoded1, predictions=cycle1_encoded2)

        with tf.variable_scope('cycle2'):

            # encoder2
            cycle2_encoded2 = self.en2_feature

            # decoder1
            cycle2_decoded = tf.add(tf.matmul(cycle2_encoded2, de1_d_w_1), de1_d_b_1)
            cycle2_reconstruction = tf.reshape(cycle2_decoded,
                                               [-1, self.input_height, self.input_width, self.input_channel_num])

            # encoder2
            cycle2_en1_conv_2d = tf.add(conv2d(cycle2_reconstruction, en1_c_w_1, kernel_stride), en1_c_b_1)
            cycle2_en1_conv_1 = tf.nn.elu(cycle2_en1_conv_2d)
            cycle2_en1_pool_1 = apply_max_pooling(cycle2_en1_conv_1, self.pooling_height_1st, self.pooling_width_1st,
                                                  self.pooling_stride_1st)
            shape = cycle2_en1_pool_1.get_shape().as_list()
            cycle2_en1_pool_1_flat = tf.reshape(cycle2_en1_pool_1, [-1, shape[1] * shape[2] * shape[3]])
            cycle2_encoded1 = tf.add(tf.matmul(cycle2_en1_pool_1_flat, share_w_1), share_b_1)

            self.loss_cycle2 = tf.losses.mean_squared_error(labels=cycle2_encoded2, predictions=cycle2_encoded1)

        with tf.variable_scope('y_predictor'):

            if self.train_phase == True:
                fc_drop_y = tf.nn.dropout(self.en1_feature, self.keep_prob)
            else:  # test
                fc_drop_y = tf.nn.dropout(self.en2_feature, self.keep_prob)

            y_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_y], 'y_w_1')
            y_b_1 = bias_variable([self.num_labels_y], 'y_b_1')
            logits_y = tf.add(tf.matmul(fc_drop_y, y_w_1), y_b_1)

            softmax_y = tf.nn.softmax(logits_y)
            pred_y = tf.argmax(softmax_y, 1)

            l2 = lambda_l2 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())


            if self.train_phase == True:
                self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(pred_y, self.y1), tf.float32))
                self.loss_y = \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=self.y1)) + l2

            else:  # test
                self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(pred_y, self.y2), tf.float32))
                self.loss_y = \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=self.y2)) + l2

            self.optimizer_y = \
                tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_y,
                                                                    var_list=[en1_c_w_1, en1_c_b_1,
                                                                              share_w_1, share_b_1,
                                                                              y_w_1, y_b_1])


        with tf.variable_scope('u_predictor_minimize'):  # in order to train user classifier

            u_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_u], 'u_w_1')
            u_b_1 = bias_variable([self.num_labels_u], 'u_b_1')

            feat_no_flip = tf.concat([self.en1_feature, self.en2_feature],0)
            self.u = tf.concat([self.u1, self.u2],0)
            # feat_no_flip = flip_gradient(self.feature, self.lambda_weight)
            fc_drop_u_no_flip = tf.nn.dropout(feat_no_flip, self.keep_prob)

            logits_u_no_flip = tf.add(tf.matmul(fc_drop_u_no_flip, u_w_1), u_b_1)

            softmax_u_no_flip = tf.nn.softmax(logits_u_no_flip)
            pred_u_no_flip = tf.argmax(softmax_u_no_flip, 1)
            self.accuracy_u_no_flip = tf.reduce_mean(tf.cast(tf.equal(pred_u_no_flip, self.u), tf.float32))

            self.loss_u_no_flip = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_u_no_flip, labels=self.u)) + l2

            self.optimizer_u_no_flip = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_u_no_flip,
                                                                                               var_list=[u_w_1, u_b_1])

        with tf.variable_scope('u_predictor_with_flip'):  # in order to train feature extractor

            feat = tf.concat([self.en1_feature, self.en2_feature], 0)
            self.u = tf.concat([self.u1, self.u2], 0)
            feat = flip_gradient(feat, self.lambda_weight)
            fc_drop_u = tf.nn.dropout(feat, self.keep_prob)
            logits_u = tf.add(tf.matmul(fc_drop_u, u_w_1), u_b_1)

            softmax_u = tf.nn.softmax(logits_u)
            pred_u = tf.argmax(softmax_u, 1)
            self.accuracy_u = tf.reduce_mean(tf.cast(tf.equal(pred_u, self.u), tf.float32))

            # train feature extractor parameters, maximize user classifier loss
            self.loss_u_flip = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_u, labels=self.u)) + l2

            self.optimizer_uflip = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_u_flip,
                                                                                       var_list=[en1_c_w_1, en1_c_b_1,
                                                                                                 share_w_1, share_b_1,
                                                                                                 en2_c_w_1, en2_c_b_1])


        self.loss_strcuture = self.loss_ae1 + self.loss_ae2 + self.loss_cycle1 + self.loss_cycle2
        self.optimizer_structure = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_strcuture,
                                                                                     var_list=[en1_c_w_1, en1_c_b_1,
                                                                                               share_w_1, share_b_1,
                                                                                               en2_c_w_1, en2_c_b_1,
                                                                                               de1_d_w_1, de1_d_b_1,
                                                                                               de2_d_w_1, de2_d_b_1])


class CNNCrossSub_act_opp(object):
    def __init__(self, input_height, input_width, input_channel_num,
                 num_labels_y, num_labels_u,
                 pooling_height_1st, pooling_width_1st, pooling_stride_1st,
                 learning_rate, lambda_weight):

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel_num = input_channel_num
        self.num_labels_y = num_labels_y
        self.num_labels_u = num_labels_u
        # pooling
        self.pooling_height_1st = pooling_height_1st # 1
        self.pooling_width_1st = pooling_width_1st  # no window: 75, with window: 15
        self.pooling_stride_1st = pooling_stride_1st  # no window: 10, with window:  3

        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight

        self._build_model()

    def _build_model(self):

        """
        CNN structure
        """

        # kernel parameter
        kernel_height_1st = self.input_height // 10
        kernel_width_1st = self.input_width
        kernel_stride = 1
        conv_channel_num = 40

        # set L2 penalty
        lambda_l2 = 0.0005

        self.X = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width], name='X')
        X_image = tf.reshape(self.X, [-1, self.input_height, self.input_width, self.input_channel_num])
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')
        self.u = tf.placeholder(tf.int64, shape=[None], name='u')
        # self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        with tf.variable_scope('feature_extractor'):

            # first CNN layer
            c_w_1 = weight_variable([kernel_height_1st, 5,
                                            self.input_channel_num, conv_channel_num],
                                           'c_w_1')
            c_b_1 = bias_variable([conv_channel_num], 'c_b_1')
            conv_2d = tf.add(conv2d(X_image, c_w_1, kernel_stride), c_b_1)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            conv_1 = tf.nn.elu(conv_2d)

            pool_1 = apply_max_pooling(conv_1, 1, 2, self.pooling_stride_1st)

            # second CNN layer
            c_w_2 = weight_variable([1, 5,
                                     conv_channel_num, conv_channel_num*2],
                                    'c_w_1')
            c_b_2 = bias_variable([conv_channel_num*2], 'c_b_2')
            conv_2d_2 = tf.add(conv2d(pool_1, c_w_2, kernel_stride), c_b_2)
            # conv_2d_bn = batch_norm_cnv_2d(conv_2d, self.train_phase)
            conv_2 = tf.nn.elu(conv_2d_2)

            pool_2 = apply_max_pooling(conv_2, 1, 2, self.pooling_stride_1st)

            # flattern the last layer of cnn
            shape = pool_2.get_shape().as_list()
            pool_2_flat = tf.reshape(pool_2, [-1, shape[1] * shape[2] * shape[3]])
            self.feature = pool_2_flat

        with tf.variable_scope('y_predictor'):

            fc_drop_y = tf.nn.dropout(pool_2_flat, self.keep_prob)

            y_w_1 = weight_variable([shape[1] * shape[2] * shape[3], self.num_labels_y], 'y_w_1')
            y_b_1 = bias_variable([self.num_labels_y], 'y_b_1')
            logits_y = tf.add(tf.matmul(fc_drop_y, y_w_1), y_b_1)

            softmax_y = tf.nn.softmax(logits_y)
            pred_y = tf.argmax(softmax_y, 1)
            self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(pred_y, self.y), tf.float32))

            l2 = lambda_l2 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss_y = \
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_y, labels=self.y)) + l2

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_y)
