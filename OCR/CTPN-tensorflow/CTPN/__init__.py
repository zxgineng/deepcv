from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim


class Graph:
    def __init__(self, mode):
        self.mode = mode

    def build(self, inputs):
        net = self.build_vgg(inputs)
        net = slim.conv2d(net,512,3,scope='rpn')
        net = tf.squeeze(net, 0)
        lstm_outputs = self.build_bilstm(net)
        vcoords_logits, scores_logits, side_logits = self.build_fc(lstm_outputs)
        return vcoords_logits, scores_logits, side_logits

    def build_vgg(self, inputs):
        with tf.variable_scope('vgg_16'):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            return net

    def build_bilstm(self, inputs):
        with tf.variable_scope('bilstm'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit,initializer=tf.orthogonal_initializer)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit,initializer=tf.orthogonal_initializer)
            lstm_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs,
                                                                          dtype=tf.float32)
            lstm_outputs = tf.concat(lstm_outputs, axis=-1)
            return lstm_outputs

    def build_fc(self, inputs):
        with tf.variable_scope('fully_connected'):
            fc = slim.fully_connected(inputs, Config.model.fc_unit)
            vcoords_logits = slim.fully_connected(fc, 2 * len(Config.model.anchor_height), activation_fn=None)
            vcoords_logits = tf.reshape(vcoords_logits,[-1,2])
            scores_logits = slim.fully_connected(fc, 2 * len(Config.model.anchor_height), activation_fn=None)
            scores_logits = tf.reshape(scores_logits,[-1,2])
            side_logits = slim.fully_connected(fc, len(Config.model.anchor_height), activation_fn=None)
            side_logits = tf.reshape(side_logits, [-1])
            return vcoords_logits, scores_logits, side_logits
