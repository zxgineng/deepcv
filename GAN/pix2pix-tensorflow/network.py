from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim


def _batch_norm(inputs):
    return slim.batch_norm(inputs, scale=True, epsilon=1e-5)


def _conv2d(inputs, out_channels, stride, activation_fn=None, normalizer_fn=None, name=None):
    with tf.variable_scope(name, "conv"):
        in_channels = inputs.shape[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        if normalizer_fn:
            conv = normalizer_fn(conv)
        if activation_fn:
            conv = activation_fn(conv)
        return conv


def _deconv(inputs, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = inputs.shape.as_list()
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(inputs, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1],
                                      padding="SAME")
        return conv


class Generator_Graph():
    def __init__(self, mode):
        self.mode = mode

    def build(self, inputs):
        net = self.encoder_phase(inputs)
        outputs = self.decoder_phase(net)
        return outputs

    def encoder_phase(self, inputs):
        num = Config.model.base_generator_filter
        with tf.variable_scope('encoder_1'):
            outputs = _conv2d(inputs, num, 2)
            tf.add_to_collection('encoder_layers', outputs)
        for i, out_channels in enumerate([num * 2, num * 4, num * 8, num * 8, num * 8, num * 8, num * 8]):
            with tf.variable_scope("encoder_%d" % (i + 2)):
                activated = tf.nn.leaky_relu(outputs)
                conv = _conv2d(activated, out_channels, 2)
                outputs = _batch_norm(conv)
                tf.add_to_collection('encoder_layers', outputs)
        return outputs

    def decoder_phase(self, inputs):
        num = Config.model.base_generator_filter
        encoder_layers = tf.get_collection('encoder_layers')
        for i, (out_channels, keep_prob) in enumerate(
                [(num * 8, 0.5), (num * 8, 0.5), (num * 8, 0.5), (num * 8, 1.0), (num * 4, 1.0), (num * 2, 1.0),
                 (num, 1.0)]):
            with tf.variable_scope("decoder_%d" % (i + 1)):
                if i == 0:
                    net = inputs
                else:
                    net = tf.concat([net, encoder_layers[7 - i]], axis=3)
                activated = tf.nn.relu(net)
                net = _deconv(activated, out_channels)
                net = _batch_norm(net)

                net = slim.dropout(net, keep_prob=keep_prob)

        with tf.variable_scope('decoder_8'):
            net = tf.concat([net, encoder_layers[0]], -1)
            activated = tf.nn.relu(net)
            outputs = _deconv(activated, 3)
            outputs = tf.nn.tanh(outputs)
        return outputs


class Discriminator_Graph():
    def __init__(self):
        pass

    def build(self, images, contours):
        inputs = tf.concat([images, contours], -1)
        num = Config.model.base_discriminator_filter
        net = _conv2d(inputs, num, 2, activation_fn=tf.nn.leaky_relu, name='layer_1')
        net = _conv2d(net, num * 2, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=_batch_norm, name='layer_2')
        net = _conv2d(net, num * 4, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=_batch_norm, name='layer_3')
        net = _conv2d(net, num * 8, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=_batch_norm, name='layer_4')
        logits = _conv2d(net, 1, 1, activation_fn=None, name='layer_5')

        return logits
