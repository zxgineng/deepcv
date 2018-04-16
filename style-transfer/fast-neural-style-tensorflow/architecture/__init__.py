from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim
from .vgg import vgg_16


class Graph:
    def __init__(self, mode):
        self.mode = mode

    def build(self, inputs):
        with tf.variable_scope('generation'):
            pad = tf.pad(inputs, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
            net = self.instance_norm_conv2d(pad, 32, 9, 1, 'conv1')
            net = self.instance_norm_conv2d(net, 64, 3, 2, 'conv2')
            net = self.instance_norm_conv2d(net, 128, 3, 2, 'conv3')
            net = self.residual_conv2d(net, 128, 3, 1, 'res1')
            net = self.residual_conv2d(net, 128, 3, 1, 'res2')
            net = self.residual_conv2d(net, 128, 3, 1, 'res3')
            net = self.residual_conv2d(net, 128, 3, 1, 'res4')
            net = self.residual_conv2d(net, 128, 3, 1, 'res5')
            net = self.resized_conv2d(net, 64, 3, 2, 'deconv1')
            net = self.resized_conv2d(net, 32, 3, 2, 'deconv2')
            net = self.instance_norm_conv2d(net, 3, 9, 1, 'deconv3', activation=tf.nn.tanh)
            outputs = (net + 1) * 127.5
            height = outputs.shape[1]
            width = outputs.shape[2]
            generated = tf.slice(outputs, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return generated, None
        else:
            processed = tf.image.resize_images(generated, [Config.model.image_size, Config.model.image_size])
            processed = processed - Config.model.channels_mean
            vgg_inputs = tf.concat([processed, inputs], 0)
            _, end_points = vgg_16(vgg_inputs)

            return generated, end_points

    def instance_norm_conv2d(self, inputs, num_outputs, kernel_size, stride, scope, mode='REFLECT',
                             activation=tf.nn.relu):
        with tf.variable_scope(scope):
            inputs = tf.pad(inputs, [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2],
                                     [0, 0]], mode=mode)
            conv = slim.conv2d(inputs, num_outputs, kernel_size, stride, 'VALID', activation_fn=None)
            outputs = slim.instance_norm(conv, activation_fn=activation)
            return outputs

    def residual_conv2d(self, inputs, num_outputs, kernel_size, stride, scope, mode='REFLECT'):
        with tf.variable_scope(scope):
            pad = tf.pad(inputs, [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2],
                                  [0, 0]], mode=mode)
            conv1 = slim.conv2d(pad, num_outputs, kernel_size, stride, 'VALID')
            conv1 = tf.pad(conv1, [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2],
                                   [0, 0]], mode=mode)
            conv2 = slim.conv2d(conv1, num_outputs, kernel_size, stride, 'VALID', activation_fn=None)
            residual = inputs + conv2
            return residual

    def resized_conv2d(self, inputs, num_outputs, kernel_size, stride, scope, mode='REFLECT', activation=tf.nn.relu):
        with tf.variable_scope(scope):
            height = inputs.shape[1]
            width = inputs.shape[2]
            new_height = height * stride * 2
            new_width = width * stride * 2
            resized = tf.image.resize_images(inputs, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            outputs = self.instance_norm_conv2d(resized, num_outputs, kernel_size, stride, 'in_conv', mode, activation)
            return outputs

