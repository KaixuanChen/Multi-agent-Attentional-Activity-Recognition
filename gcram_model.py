import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
from functions import *

# get worse results than grcam_model


def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def _log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    locs = tf.stack(locs)
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)  # [batch_sz, timesteps]


class RetinaSensor(object):
    # one scale
    def __init__(self, img_size_width, img_size_height, patch_window_width, patch_window_height):
        self.img_size_width = img_size_width
        self.img_size_height = img_size_height
        self.patch_window_width = patch_window_width
        self.patch_window_height = patch_window_height

    def __call__(self, img_ph, loc):
        img = tf.reshape(img_ph, [
            tf.shape(img_ph)[0],
            self.img_size_width,
            self.img_size_height,
            1
        ])
        '''
        tf.image.extract_glimpse:
        If the windows only partially
        overlaps the inputs, the non overlapping areas will be filled with
        random noise.
        '''
        pth = tf.image.extract_glimpse(
            img, # input
            [self.patch_window_width, self.patch_window_height], # size
            loc) # offset
        # pth: [tf.shape(img_ph)[0], patch_window_width, patch_window_height, 1]

        return tf.reshape(pth, [tf.shape(loc)[0],
                                self.patch_window_height,  self.patch_window_width])


class GlimpseNetwork(object):
    def __init__(self, img_size_width, img_size_height,
                 patch_window_width, patch_window_height,
                 select_dim, g_size, l_size, output_size, nb_locations):
        self.retina_sensor = RetinaSensor(img_size_width, img_size_height,
                                          patch_window_width, patch_window_height)
        self.cnn = CNN(patch_window_height, nb_locations*patch_window_width)

        # layer 1
        self.g1_w = _weight_variable((200, g_size)) # 200: cnn results
        self.g1_b = _bias_variable((g_size,))
        self.l1_w = _weight_variable((4, l_size)) # 4: 3 locs + 1 t
        self.l1_b = _bias_variable((l_size,))
        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))
        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, imgs_ph, loc_1, loc_2, loc_3, t):
        pths_1 = self.retina_sensor(imgs_ph, tf.concat([loc_1,t], 1))
        pths_2 = self.retina_sensor(imgs_ph, tf.concat([loc_2,t], 1))
        pths_3 = self.retina_sensor(imgs_ph, tf.concat([loc_3,t], 1))

        whole_patch = tf.concat([pths_1, pths_2, pths_3], 2)
        whole_patch_CNN = self.cnn(whole_patch)

        g_1 = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(whole_patch_CNN, self.g1_w, self.g1_b)),
                            self.g2_w, self.g2_b)
        l_1 = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(tf.concat([loc_1, loc_2, loc_3, t], 1), self.l1_w, self.l1_b)),
                            self.l2_w, self.l2_b)

        return tf.nn.relu(g_1 + l_1)


class LocationNetwork(object):
    def __init__(self, loc_dim, rnn_output_size, variance=0.22, is_sampling=False):
        self.loc_dim = loc_dim  # 2, (x,y)
        self.variance = variance
        self.w = _weight_variable((rnn_output_size, loc_dim))
        self.b = _bias_variable((loc_dim,))

        self.is_sampling = is_sampling

    def __call__(self, cell_output):
        mean = tf.nn.xw_plus_b(cell_output, self.w, self.b)
        mean = tf.clip_by_value(mean, -1., 1.)
        mean = tf.stop_gradient(mean)

        if self.is_sampling:
            loc = mean + tf.random_normal(
                (tf.shape(cell_output)[0], self.loc_dim),
                stddev=self.variance)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean

class CNN(object):
    def __init__(self, input_height, input_width):
        self.input_height = input_height
        self.input_width = input_width
        kernel_height_1st = 1  # self.input_height
        kernel_width_1st = self.input_width
        input_channel_num = 1
        conv_channel_num_1st = 40
        self.kernel_stride = 1
        self.c_w_1 = _weight_variable((kernel_height_1st, kernel_width_1st,
                                  input_channel_num, conv_channel_num_1st)
                                 )
        self.c_b_1 = _bias_variable((conv_channel_num_1st,))

    def __call__(self, imgs_ph):
        imgs_ph_re = tf.reshape(imgs_ph, [-1, self.input_height, self.input_width, 1])


        conv_2d_1st = tf.nn.relu(tf.add(conv2d(imgs_ph_re, self.c_w_1, self.kernel_stride), self.c_b_1))
        shape = conv_2d_1st.get_shape().as_list()
        conv_2d_flat = tf.reshape(conv_2d_1st, [-1, shape[1]*shape[2]*shape[3]])

        return conv_2d_flat

class GlobalConvolutionalRecurrentAttentionModel(object):
    def __init__(self, img_width, img_height, nb_locations,
                 glimpse_width, glimpse_height,
                 g_size, l_size, glimpse_output_size, loc_dim, time_dim, variance,
                 cell_size, nb_glimpses, nb_classes, learning_rate, learning_rate_decay_factor,
                 min_learning_rate, nb_training_batch, max_gradient_norm, is_training=False):

        self.img_ph = tf.placeholder(tf.float32, [None, img_height, img_width])
        self.lbl_ph = tf.placeholder(tf.int64, [None])

        self.global_step = tf.Variable(0, trainable=False)
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / training_batch_num)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            nb_training_batch, # batch number
            learning_rate_decay_factor,
            # If the argument staircase is True,
            # then global_step / decay_steps is an integer division
            # and the decayed learning rate follows a staircase function.
            staircase=True),
            min_learning_rate)

        cell = BasicLSTMCell(cell_size)


        with tf.variable_scope('GlimpseNetwork'):
            glimpse_network = GlimpseNetwork(img_width,
                                             img_height,
                                             glimpse_width,
                                             glimpse_height,
                                             loc_dim+time_dim,
                                             g_size,
                                             l_size,
                                             glimpse_output_size,
                                             nb_locations)
        with tf.variable_scope('LocationNetwork'):
            location_network = LocationNetwork(loc_dim=loc_dim*nb_locations+time_dim,
                                               rnn_output_size=cell.output_size, # cell_size
                                               variance=variance,
                                               is_sampling=is_training)

        # with tf.variable_scope('CNN'):
        #     cnn = CNN(nb_locations, glimpse_output_size)

        # with tf.variable_scope('CDD'):
        #     cdd = CDD(glimpse_height, nb_locations*glimpse_output_size)

        # Core Network
        batch_size = tf.shape(self.img_ph)[0]
        init_loc_1 = tf.random_uniform((batch_size, loc_dim), minval=-1, maxval=1)
        init_loc_2 = tf.random_uniform((batch_size, loc_dim), minval=-1, maxval=1)
        init_loc_3 = tf.random_uniform((batch_size, loc_dim), minval=-1, maxval=1)
        init_t = tf.random_uniform((batch_size, loc_dim), minval=-1, maxval=1)
        # shape: (batch_size, loc_dim), range: [-1,1)
        init_state = cell.zero_state(batch_size, tf.float32)

        self.init_glimpse = glimpse_network(self.img_ph, init_loc_1, init_loc_2, init_loc_3,
                                                                         init_t)
        # self.init_glimpse_cooperate = cnn(self.init_glimpse)

        # self.imgs_ph, self.imgs_ph_re, self.h_fc1, self.conv_2d_1st, self.conv_2d_2nd, self.conv_2d_flat = cdd(self.init_glimpse)

        rnn_inputs = [self.init_glimpse]
        rnn_inputs.extend([0] * nb_glimpses)

        locs, loc_means = [], []

        def loop_function(prev, _):
            loc, loc_mean = location_network(prev)
            locs.append(loc)
            loc_means.append(loc_mean)
            glimpse = glimpse_network(self.img_ph, tf.reshape(loc[:,0],[-1,1]),
                                      tf.reshape(loc[:, 1], [-1, 1]),
                                      tf.reshape(loc[:, 2], [-1, 1]),
                                      tf.reshape(loc[:, 3], [-1, 1]))
            # glimpse_cooperate = cnn(glimpse)
            return glimpse

        rnn_outputs, _ = rnn_decoder(rnn_inputs, init_state, cell, loop_function=loop_function)

        # Time independent baselines
        with tf.variable_scope('Baseline'):
            baseline_w = _weight_variable((cell.output_size, 1))
            baseline_b = _bias_variable((1,))
        baselines = []
        for output in rnn_outputs[1:]:
            baseline = tf.nn.xw_plus_b(output, baseline_w, baseline_b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)
        baselines = tf.stack(baselines)  # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

        # Classification. Take the last step only.
        rnn_last_output = rnn_outputs[-1]
        with tf.variable_scope('Classification'):
            logit_w = _weight_variable((cell.output_size, nb_classes))
            logit_b = _bias_variable((nb_classes,))
        logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)
        # self.prediction = tf.argmax(logits, 1)
        self.softmax = tf.nn.softmax(logits)

        self.pred = tf.argmax(self.softmax, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.lbl_ph), tf.float32))


        if is_training:
            # classification loss
            self.cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
            # RL reward
            reward = tf.cast(tf.equal(self.pred, self.lbl_ph), tf.float32)
            rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
            rewards = tf.tile(rewards, (1, nb_glimpses))  # [batch_sz, timesteps]
            advantages = rewards - tf.stop_gradient(baselines)
            self.advantage = tf.reduce_mean(advantages)
            logll = _log_likelihood(loc_means, locs, variance)
            logllratio = tf.reduce_mean(logll * advantages)
            self.reward = tf.reduce_mean(reward)
            # baseline loss
            self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
            # hybrid loss

            self.loss = -logllratio + self.cross_entropy + self.baselines_mse
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

